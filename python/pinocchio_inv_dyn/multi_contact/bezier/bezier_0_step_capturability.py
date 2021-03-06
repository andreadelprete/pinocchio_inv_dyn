# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:54:39 2016

@author: stonneau
"""

from pinocchio_inv_dyn.optimization.solver_LP_abstract import LP_status, LP_status_string
from pinocchio_inv_dyn.multi_contact.stability_criterion import  Bunch
from pinocchio_inv_dyn.optimization.solver_LP_abstract import getNewSolver
from pinocchio_inv_dyn.abstract_solver import AbstractSolver as qp_solver

from spline import bezier, bezier6, polynom, bernstein

from numpy import array, vstack, zeros, ones, sqrt, matrix, asmatrix, asarray, identity
from numpy import cross as X
from numpy.linalg import norm
import numpy as np
from math import atan, pi, sqrt

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
    
def splitId(v):
    dim = v.shape[0]
    res = zeros([dim,dim]);
    for i in range(dim):
        res[i,i] = v[i]
    return res
        

def __init_6D():
    return zeros([6,3]), zeros(6)
    



def normalize(A,b=None):
    null_rows = []
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
def compute_CWC(p, N, mass, mu, T=None):    
    __cg = 4; #num generators per contact
    eq = Equilibrium("dyn_eq2", mass, __cg) 
    if T == None:
        eq.setNewContacts(asmatrix(p),asmatrix(N),mu,EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_PP)
    else:
        eq.setNewContactsWithTangents(asmatrix(p),asmatrix(N),asmatrix(T),mu,EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_PP)
    H, h = eq.getPolytopeInequalities()
    assert(norm(h) < __EPS), "h is not equal to zero"
    return  normalize(np.squeeze(np.asarray(-H)))

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
    
#angular momentum waypoints
def u0(l0, alpha):
    ux, us = __init_6D()    
    ux[3:] = identity(3)* 3 * alpha
    us[3:] = -3*alpha*l0[:]
    return  (ux, us)
    
def u1(l0, alpha):
    ux, us = __init_6D()  
    us[3:] = -1.5*l0*alpha
    return  (ux, us)
    
def u2(l0, alpha):
    ux, us = __init_6D()    
    ux[3:] = identity(3)* (-1.5) * alpha
    us[3:] = -l0 / 2. * alpha
    return  (ux, us)
    
def u3(l0, alpha):
    ux, us = __init_6D()    
    ux[3:] = identity(3)*  (-1.5) * alpha
    return  (ux, us)
    
def u4(l0, alpha):
    ux, us = __init_6D()   
    return  (ux, us)


wis = [w0,w1,w2,w3,w4]
uis = [u0,u1,u2,u3,u4]
b4 = [bernstein(4,i) for i in range(5)]

      
def c_of_t(curve, T):
    def _eval(t):
        return  asarray(curve(t/T)).flatten()
    return _eval
def dc_of_t(curve, T):
    def _eval(t):
        return  1/T * asarray(curve(t/T)).flatten()
    return _eval
def ddc_of_t(curve, T):
    def _eval(t):
        return  1/(T*T) * asarray(curve(t/T)).flatten()
    return _eval

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
    
    def __init__ (self, name, c0, dc0, contact_points, contact_normals, mu, g, mass, kinematic_constraints = None, angular_momentum_constraints = None,
                  contactTangents = None, maxIter=1000, verb=0, regularization=1e-5, solver='qpoases'):
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
        self.set_contacts(contact_points, contact_normals, mu, contactTangents)
        if kinematic_constraints != None:
            self._kinematic_constraints = kinematic_constraints[:]
        else:
            self._kinematic_constraints = None        
        if angular_momentum_constraints != None:
            self._angular_momentum_constraints = angular_momentum_constraints[:]
        else:
            self._angular_momentum_constraints = None   
                 
        self._lp_solver = getNewSolver('qpoases', "name", useWarmStart=False, verb=0)
        self._qp_solver = qp_solver(3, 0, solver='qpoases', accuracy=1e-6, maxIter=100, verb=0)
        
    def init_bezier(self, c0, dc0, n, T =1.):
        self._n = n
        self._p0 = c0[:]
        self._p1 = dc0 * T / n +  self._p0  
        self._p0X = skew(c0)
        self._p1X = skew(self._p1)

    def set_contacts(self, contact_points, contact_normals, mu, contactTangents):
        self._contact_points    = np.asarray(contact_points).copy();
        self._contact_normals   = np.asarray(contact_normals).copy();
        self._mu                = mu;
        self._H                 = compute_CWC(self._contact_points, self._contact_normals, self._mass, mu, contactTangents)#CWC inequality matrix
                       
                          
    def __compute_wixs(self, T, num_step = -1):        
        alpha = 1. / (T*T)
        wps = [wi(self._p0, self._p1, self._g, self._p0X, self._p1X, self._gX, alpha)  for wi in wis]
        if num_step > 0:
            dt = (1./float(num_step))
            wps_bern = [ [ (b(i*dt)*wps[idx][0], b(i*dt)*wps[idx][1]) for idx,b in enumerate(b4)] for i in range(num_step + 1) ]
            wps = [reduce(lambda a, b : (a[0] + b[0], a[1] + b[1]), wps_bern_i) for wps_bern_i in wps_bern]
        return wps
        
    #angular momentum waypoints
    def __compute_uixs(self, l0, T, num_step = -1):        
        alpha = 1. / (T)
        wps = [ui(l0, alpha)  for ui in uis]
        if num_step > 0:
            dt = (1./float(num_step))
            wps_bern = [ [ (b(i*dt)*wps[idx][0], b(i*dt)*wps[idx][1]) for idx,b in enumerate(b4)] for i in range(num_step + 1) ]
            wps = [reduce(lambda a, b : (a[0] + b[0], a[1] + b[1]), wps_bern_i) for wps_bern_i in wps_bern]
        return wps

    def _init_matrices_AL_bL(self, ups, A, b):    
        dimL = 0
        if self._angular_momentum_constraints != None:
            dimL = self._angular_momentum_constraints[0].shape[0]        
        AL = zeros([A.shape[0]+dimL, 6]); 
        bL = zeros([A.shape[0]+dimL   ]); 
        AL[:A.shape[0],:3] = A[:]
        bL[:b.shape[0]   ] = b[:] 
        return AL,bL
        
    def __add_angular_momentum(self,A,b,l0, T, num_steps):       
        ups = self.__compute_uixs(l0, T ,num_steps)    
        AL, bL = self._init_matrices_AL_bL(ups, A, b)      
        dimH  = self._H.shape[0]          
        #final matrix has num rows equal to initial matrix rows + angular momentum constraints
        # the angular momentum constraints are added AFTER the eventual kinematic ones
        for i, (uxi, usi) in enumerate(ups):       
            AL[i*dimH : (i+1)*dimH, 3:]  = self._H.dot(uxi) #constant part of A, Ac = Ac * wxi
            bL[i*dimH : (i+1)*dimH    ] += self._H.dot(-usi)  
        
        if self._angular_momentum_constraints != None:
            dimL = self._angular_momentum_constraints[0].shape[0]
            AL[-dimL:,3:] = self._angular_momentum_constraints[0][:]
            bL[-dimL:   ] = self._angular_momentum_constraints[1][:]
        
        AL, bL = normalize(AL,bL)
        return AL, bL
    
    def __add_kinematic_and_normalize(self,A,b, norm = True):        
        if self._kinematic_constraints != None:
            dim_kin = self._kinematic_constraints[0].shape[0]
            A[-dim_kin:,:] = self._kinematic_constraints[0][:]
            b[-dim_kin:] =  self._kinematic_constraints[1][:]
        if(norm):
            A, b = normalize(A,b)
        return A, b
        
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
        
    
    def compute_6d_control_point_inequalities(self, T, time_step = -1., l0 = None):
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
        use_angular_momentum = l0 != None
        A,b = self.__add_kinematic_and_normalize(A,b, not use_angular_momentum)
        if use_angular_momentum:
            A,b = self.__add_angular_momentum(A,b, l0, T, num_steps)        
        self.__Ain = A[:]; self.__Aub = b[:]
        
            
    
    def _solve(self, dim_pb, l0, asLp = False, guess = None ):
        cost = 0
        if asLp:
            c = zeros(dim_pb); c[2] = -1
            (status, x, y) = self._lp_solver.solve(c, lb= -100. * ones(dim_pb), ub = 100. * ones(dim_pb),
                                                       A_in=self.__Ain, Alb=-100000.* ones(self.__Ain.shape[0]), Aub=self.__Aub,
                                                       A_eq=None, b=None)
            return status, x, cost, self._lp_solver.getLpTime()
        else:
            #~ self._qp_solver = qp_solver(dim_pb, self.__Ain.shape[0], solver='qpoases', accuracy=1e-6, maxIter=100, verb=0)
            self._qp_solver.changeInequalityNumber(self.__Ain.shape[0], dim_pb)
            #weight_dist_or = 0.001
            weight_dist_or = 0
            D = identity(dim_pb); 
            alpha = sqrt(12./5.) 
            for i in range(3):
                D[i,i] = weight_dist_or
            d = zeros(dim_pb);
            d[:3]= self._p0 * weight_dist_or
            if(l0 != None):
                # minimizing integral of angular momentum 
                for i in range(3,6):
                    D[i,i] = alpha
                d[3:]= (9.* l0) / (5. * alpha)
            D = (D[:]); d = (d[:]); A = (self.__Ain[:]);
            lbA = (-100000.* ones(self.__Ain.shape[0]))[:]; ubA=(self.__Aub);
            lb = (-100. * ones(dim_pb))[:]; ub = (100. * ones(dim_pb))[:];    
            self._qp_solver.setProblemData(D = D , d = d, A=A, lbA=lbA, ubA=ubA, lb = lb, ub = ub, x0=None)
            (x, imode) =  self._qp_solver.solve(D = D , d = d, A=A, lbA=lbA, ubA=ubA, lb = lb, ub = ub, x0=None)
            if l0 == None:
                cost = norm(self._p0 - x)
            else:
                cost = (1./5.)*(9.*l0.dot(l0) -  9.*l0.dot(x[3:]) + 6.*x[3:].dot(x[3:]))
            return imode, x, cost , self._qp_solver.qpTime
        
    def can_I_stop(self, c0=None, dc0=None, T=1., MAX_ITER=None, time_step = -1, l0 = None, asLp = False):
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
              l0 : if equals None, angular momentum is not considered and set to 0. Else
              it becomes a variable of the problem and l0 is the initial angular momentum
              asLp : If true, problem is solved as an LP. If false, solved as a qp that
              minimizes distance to original point (weight of 0.001) and angular momentum if applies (weight of 1.)
            Output: An object containing the following member variables:
              is_stable -- boolean value
              c -- final com position
              dc -- final com velocity. [WARNING] if is_stable is False, not used
              ddc_min -- [WARNING] Not relevant (used)
              t -- always T (Bezier curve)
              computation_time -- time taken to solve all the LPs
              c_of_t, dc_of_t, ddc_of_t: trajectories and derivatives in function of the time
              dL_of_t : trajectory of the angular momentum along time
              wps  : waypoints of the solution bezier curve c*(s)
              wpsL : waypoints of the solution angular momentum curve   L*(s) Zero if no angular mementum
              wpsdL : waypoints of the solution angular momentum curve dL*(s) Zero if no angular mementum
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
        use_angular_momentum = l0 != None
        # for the moment c is random stuff.
        dim_pb = 6 if use_angular_momentum else 3
        c = zeros(dim_pb); c[2] = -1
        wps = self.compute_6d_control_point_inequalities(T, time_step, l0) 
        status, x, cost, comp_time = self._solve(dim_pb, l0, asLp)
        is_stable=status==LP_status.LP_STATUS_OPTIMAL
        wps   = [self._p0,self._p1,x[:3],x[:3]]; 
        wpsL  = [zeros(3) if not use_angular_momentum else l0[:], zeros(3) if not use_angular_momentum else x[-3:] ,zeros(3),zeros(3)]; 
        wpsdL = [3*(wpsL[1] - wpsL[0]) ,3*(- wpsL[1]), zeros(3)]; 
        c_of_s = bezier(matrix([pi.tolist() for pi in wps]).transpose())       
        dc_of_s  = c_of_s.compute_derivate(1)
        ddc_of_s = c_of_s.compute_derivate(2)
        dL_of_s  = bezier(matrix([pi.tolist() for pi in wpsdL]).transpose())         
        L_of_s  = bezier(matrix([pi.tolist() for pi in wpsL]).transpose())         
        
        return Bunch(is_stable=is_stable, c=x[:3], dc=zeros(3), 
                             computation_time = comp_time, ddc_min=0.0, t = T, 
                             c_of_t = c_of_t(c_of_s, T), dc_of_t = dc_of_t(dc_of_s, T), ddc_of_t = c_of_t(ddc_of_s, T), dL_of_t = dc_of_t(dL_of_s, T), L_of_t = c_of_t(L_of_s, T),
                              cost = cost, wps = wps, wpsL = wpsL,  wpsdL = wpsdL);

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
