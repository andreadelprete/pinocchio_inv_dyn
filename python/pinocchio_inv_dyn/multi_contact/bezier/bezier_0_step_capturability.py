# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:54:39 2016

@author: stonneau
"""

from pinocchio_inv_dyn.optimization.solver_LP_abstract import LP_status, LP_status_string
from pinocchio_inv_dyn.multi_contact.stability_criterion import  Bunch
from pinocchio_inv_dyn.optimization.solver_LP_abstract import getNewSolver

from spline import bezier, bezier6, polynom, bernstein

from numpy import array, vstack, zeros, ones, sqrt, matrix, asmatrix, asarray, identity
from numpy import cross as X
from numpy.linalg import norm
import numpy as np
from math import atan, pi

import cProfile

np.set_printoptions(precision=2, suppress=True, linewidth=100);

from centroidal_dynamics import *

__EPS = 1e-8;



def skew(x):
    res = zeros([3,3])
    res[0,0] =     0;  res[0,1] = -x[2]; res[0,2] =  x[1];
    res[1,0] =  x[2];  res[1,1] =    0 ; res[1,2] = -x[0];
    res[2,0] = -x[1];  res[2,1] =  x[0]; res[2,2] =   0  ;
    return res
        

def __init_6D():
    return zeros([6,3]), zeros(6)



def normalize(A,b=None):
    for i in range (A.shape[0]):
        n_A = norm(A[i,:])
        if(n_A != 0.):
            A[i,:] = A[i,:] / n_A
            if b != None:
                b[i] = b[i] / n_A
    if b== None:
        return A
    return A, b

## 
#  Given a list of contact points
#  as well as a list of associated normals
#  compute the gravito inertial wrench cone
#  \param p array of 3d contact positions
#  \param N array of 3d contact normals
#  \param mass mass of the robot
#  \param mu friction coefficient
#  \return the CWC H, H w <= 0, where w is the wrench. [WARNING!] TODO: The H matrix is such that 
#  the wrench w is the one present in the ICRA paper 15 of del prete et al., contrary to the current c++ implementation
def compute_CWC(p, N, mass, mu):    
    __cg = 4; #num generators per contact
    eq = Equilibrium("dyn_eq2", mass, __cg) 
    eq.setNewContacts(asmatrix(p),asmatrix(N),mu,EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_PP)
    H, h = eq.getPolytopeInequalities()
    assert(norm(h) < __EPS), "h is not equal to zero"
    #~ print "len H, ", H.shape[0]
    return normalize(np.squeeze(np.asarray(-H)))

#################################################
# global constant bezier variables and methods ##
#################################################
def w0(p0, p1, g, p0X, p1X, gX, alpha):
    wx, ws = __init_6D()    
    wx[:3,:] = 6*alpha*identity(3);  wx[3:,:] = 6*alpha*p0X;
    ws[:3]   = 6*alpha*(p0 - 2*p1) 
    ws[3:]   = X(-p0, 12*alpha*p1 + g ) 
    return  (wx, ws)
    
def w1(p0, p1, g, p0X, p1X, gX, alpha):
    wx, ws = __init_6D()    
    wx[:3,:] = 3*alpha*identity(3);  
    wx[3:,:] = skew(1.5 * (3*p1 - p0))*alpha
    ws[:3]   =  1.5 *alpha* (3*p0 - 5*p1);
    ws[3:]   = X(3*alpha*p0, -p1) + 0.25 * (gX.dot(3*p1 + p0)) 
    return  (wx, ws)
    
def w2(p0, p1, g, p0X, p1X, gX, alpha):
    wx, ws = __init_6D()    
    #~ wx[:3,:] = 0;  
    wx[3:,:] = skew(0.5*g - 3*alpha* p0 + 3*alpha*p1)
    ws[:3]   =  3*alpha*(p0 - p1);
    ws[3:]   = 0.5 * gX.dot(p1) 
    return  (wx, ws)
    
def w3(p0, p1, g, p0X, p1X, gX, alpha):
    wx, ws = __init_6D()    
    wx[:3,:] = -3*alpha* identity(3);  
    wx[3:,:] = skew(g - 1.5 *alpha* (p1 + p0))
    ws[:3]   = 1.5*alpha * (p1 + p0) 
    #~ ws[3:]   = 0 
    return  (wx, ws)
    
def w4(p0, p1, g, p0X, p1X, gX, alpha):
    wx, ws = __init_6D()    
    wx[:3,:] = -6*alpha *identity(3);  
    wx[3:,:] = skew(g - 6*alpha* p1)
    ws[:3]   = 6*alpha*p1 
    #~ ws[3:]   = 0 
    return  (wx, ws)
    

wis = [w0,w1,w2,w3,w4]
b4 = [bernstein(4,i) for i in range(5)]

#################################################
# BezierZeroStepCapturability                  ##
#################################################
class BezierZeroStepCapturability(object):
    _name = ""
    _maxIter = 0;
    _verb = 0;
    _com_acc_solver = None;
    _c0 = None;
    _dc0 = None;
    
    _computationTime = 0.0;
    _outerIterations = 0;
    _innerIterations = 0;
    
    def __init__ (self, name, c0, dc0, contact_points, contact_normals, mu, g, mass, kinematic_constraints = None, 
                  maxIter=1000, verb=0, regularization=1e-5, solver='qpoases'):
        ''' Constructor
            @param c0 Initial CoM position
            @param dc0 Initial CoM velocity
            @param contact points A matrix containing the contact points
            @param contact normals A matrix containing the contact normals
            @param mu Friction coefficient (either a scalar or an array)
            @param g Gravity vector
            @param mass The robot mass            
            @param kinematic constraints couple [A,b] such that the com is constrained by  A x <= b
            @param regularization Weight of the force minimization, the higher this value, the sparser the solution
        '''
        assert mass>0.0, "Mass is not positive"
        assert mu>0.0, "Friction coefficient is not positive"
        assert np.asarray(c0).squeeze().shape[0]==3, "Com position vector has not size 3"
        assert np.asarray(dc0).squeeze().shape[0]==3, "Com velocity vector has not size 3"
        assert np.asarray(contact_points).shape[1]==3, "Contact points have not size 3"
        assert np.asarray(contact_normals).shape[1]==3, "Contact normals have not size 3"
        assert np.asarray(contact_points).shape[0]==np.asarray(contact_normals).shape[0], "Number of contact points do not match number of contact normals"
        self._name              = name;
        self._maxIter           = maxIter;
        self._verb              = verb;
        self._c0                = np.asarray(c0).squeeze().copy();
        self._dc0               = np.asarray(dc0).squeeze().copy();
        self._mass              = mass;
        self._g                 = np.asarray(g).squeeze().copy();
        self._gX  = skew(self._g )
#        self._regularization    = regularization;
        self.set_contacts(contact_points, contact_normals, mu)
        if kinematic_constraints != None:
            self._kinematic_constraints = kinematic_constraints[:]
        else:
            self._kinematic_constraints = None        
        self._solver = getNewSolver('qpoases', "name", useWarmStart=False, verb=0)
        
    def init_bezier(self, c0, dc0, n, T =1.):
        self._n = n
        self._p0 = c0[:]
        self._p1 = dc0 * T / n +  self._p0  
        self._p0X = skew(c0)
        self._p1X = skew(self._p1)

    def set_contacts(self, contact_points, contact_normals, mu):
        self._contact_points    = np.asarray(contact_points).copy();
        self._contact_normals   = np.asarray(contact_normals).copy();
        self._mu                = mu;
        self._H                 = compute_CWC(self._contact_points, self._contact_normals, self._mass, mu)#CWC inequality matrix
                       
        
    def __compute_wixs(self, T, num_step = -1):        
        alpha = 1. / (T*T)
        wps = [wi(self._p0, self._p1, self._g, self._p0X, self._p1X, self._gX, alpha)  for wi in wis]
        if num_step > 0:
            #~ print " discrete case, num steps :", num_step
            dt = (1./float(num_step))
            wps_bern = [ [ (b(i*dt)*wps[idx][0], b(i*dt)*wps[idx][1]) for idx,b in enumerate(b4)] for i in range(num_step + 1) ]
            wps = [reduce(lambda a, b : (a[0] + b[0], a[1] + b[1]), wps_bern_i) for wps_bern_i in wps_bern]
        return wps
        
    def __add_kinematic_and_normalize(self,A,b):        
        if self._kinematic_constraints != None:
            dim_kin = self._kinematic_constraints[0].shape[0]
            A[-dim_kin:,:] = self._kinematic_constraints[0][:]
            b[-dim_kin:] =  self._kinematic_constraints[1][:]
        A, b = normalize(A,b)
        self.__Ain = A[:]; self.__Aub = b[:]
        #~ print "len constraints: ", b.shape
        #~ pass
        
    def _compute_num_steps(self, T, time_step):    
        num_steps = -1    
        if(time_step > 0.):
            num_steps = int(T / time_step)
        return num_steps
        
    def _init_matrices_A_b(self, wps):    
        dim_kin = 0
        dimH  = self._H.shape[0]
        if self._kinematic_constraints != None:
            dim_kin = self._kinematic_constraints[0].shape[0]
        A = zeros([dimH * len(wps)+dim_kin,3]) 
        b = zeros(dimH * len(wps)+ dim_kin)
        return A,b
        
    
    def compute_6d_control_point_inequalities(self, T, time_step = -1.):
        ''' compute the inequality methods that determine the 6D bezier curve w(t)
            as a function of a variable waypoint for the 3D COM trajectory.
            The initial curve is of degree 3 (init pos and velocity, 0 velocity constraints + one free variable).
            The 6d curve is of degree 2*n-2 = 4, thus 5 control points are to be computed. 
            Each control point produces a 6 * 3 inequality matrix wix, and a 6 *1 column right member wsi.
            Premultiplying it by H gives mH w_xi * x <= mH_wsi where m is the mass
            Stacking all of these results in a big inequality matrix A and a column vector x that determines the constraints
            On the 6d curves, Ain x <= Aub
        '''        
        self.init_bezier(self._c0, self._dc0, 3, T)
        dimH  = self._H.shape[0]
        mH    = self._mass *self._H         
        num_steps = self._compute_num_steps(T, time_step) 
        wps = self.__compute_wixs(T ,num_steps)
        A,b = self._init_matrices_A_b(wps)
        bc = np.concatenate([self._g,zeros(3)])  #constant part of Aub, Aubi = mH * (bc - wsi)
        for i, (wxi, wsi) in enumerate(wps):       
            A[i*dimH : (i+1)*dimH, : ]  = mH.dot(wxi) #constant part of A, Ac = Ac * wxi
            b[i*dimH : (i+1)*dimH    ]  = mH.dot(bc - wsi)
        self.__add_kinematic_and_normalize(A,b)
    
        
        
    def can_I_stop(self, c0=None, dc0=None, T=1., MAX_ITER=None, time_step = -1):
        ''' Determine whether the system can come to a stop without changing contacts.
            Keyword arguments:
              c0 -- initial CoM position 
              dc0 -- initial CoM velocity              
              T -- the EXACT given time to stop
              time_step -- if negative, a continuous resolution is used 
              to guarantee that the trajectory is feasible. If > 0, then
              used a discretized approach to validate trajectory. This allows
              to have control points outside the cone, which is supposed to increase the 
              solution space.
            Output: An object containing the following member variables:
              is_stable -- boolean value
              c -- final com position
              dc -- final com velocity. [WARNING] if is_stable is False, not used
              ddc_min -- [WARNING] Not relevant (used)
              t -- always T (Bezier curve)
              computation_time -- time taken to solve all the LPs
        '''        
        if T <=0.:
            raise ValueError('T cannot be lesser than 0')
            print "\n *** [WARNING] In bezier step capturability: you set a T_0 or MAX_ITER value, but they are not used by the algorithm"
        if MAX_ITER !=None:
            print "\n *** [WARNING] In bezier step capturability: you set a T_0 or MAX_ITER value, but they are not used by the algorithm"
        if(c0 is not None):
            assert np.asarray(c0).squeeze().shape[0]==3, "CoM has not size 3"
            self._c0 = np.asarray(c0).squeeze().copy();
        if(dc0 is not None):
            assert np.asarray(dc0).squeeze().shape[0]==3, "CoM velocity has not size 3"
            self._dc0 = np.asarray(dc0).squeeze().copy();
        if((c0 is not None) or (dc0 is not None)):
            init_bezier(self._c0, self._dc0, self._n)
                    
        ''' Solve the linear program
         minimize    c' x
         subject to  Alb <= A_in x <= Aub
                     A_eq x = b
                     lb <= x <= ub
        Return a tuple containing:
            status flag
            primal solution
            dual solution
                '''
        # for the moment c is random stuff
        c = zeros(3);c[2] = -1.
        wps = self.compute_6d_control_point_inequalities(T, time_step)
        
        #~ self._solver = getNewSolver('qpoases', "name", useWarmStart=False, verb=0)
        (status, x, y) = self._solver.solve(c, lb= -100. * ones(3), ub = 100. * ones(3), A_in=self.__Ain, Alb=-100000.* ones(self.__Ain.shape[0]), Aub=self.__Aub, A_eq=None, b=None)
        
                  
        wps = [self._p0,self._p1,x,x]; wps = matrix([pi.tolist() for pi in wps]).transpose()
        c_of_s = bezier(wps)
        dc_of_s = c_of_s.compute_derivate(1)
        ddc_of_s = c_of_s.compute_derivate(2)
        def c_of_t(curve):
            def _eval(t):
                return  asarray(curve(t/T)).flatten()
            return _eval
        def dc_of_t(curve):
            def _eval(t):
                return  1/T * asarray(curve(t/T)).flatten()
            return _eval
        def ddc_of_t(curve):
            def _eval(t):
                return  1/(T*T) * asarray(curve(t/T)).flatten()
            return _eval
        
        
        return Bunch(is_stable=status==LP_status.LP_STATUS_OPTIMAL, c=x, dc=zeros(3), 
                             computation_time=-1, ddc_min=0.0, t = T, c_of_t = c_of_t(c_of_s), dc_of_t = dc_of_t(dc_of_s), ddc_of_t = c_of_t(ddc_of_s));

    def predict_future_state(self, t_pred, c0=None, dc0=None, MAX_ITER=1000):
        ''' Compute what the CoM state will be at the specified time instant if the system
            applies maximum CoM deceleration parallel to the current CoM velocity
            Keyword arguments:
            t_pred -- Future time at which the prediction is made
            c0 -- initial CoM position 
            dc0 -- initial CoM velocity 
            Output: An object with the following member variables:
            t -- time at which the integration has stopped (equal to t_pred, unless something went wrong)
            c -- final com position
            dc -- final com velocity
        '''
        raise NotImplementedError('predict_future_state is not implemted so far.')    
