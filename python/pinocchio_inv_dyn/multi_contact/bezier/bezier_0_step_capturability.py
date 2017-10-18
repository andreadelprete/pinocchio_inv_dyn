# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:54:39 2016

@author: stonneau
"""

from pinocchio_inv_dyn.optimization.solver_LP_abstract import LP_status, LP_status_string
from pinocchio_inv_dyn.sot_utils import qpOasesSolverMsg
from pinocchio_inv_dyn.multi_contact.stability_criterion import  Bunch, EPS
from pinocchio_inv_dyn.optimization.solver_LP_abstract import getNewSolver

from spline import bezier, bezier6, polynom

from numpy import array, vstack, zeros, ones, sqrt, matrix, asmatrix, asarray, identity
from numpy import cross as X
from numpy.linalg import norm
import numpy as np
from math import atan, pi

import cProfile

np.set_printoptions(precision=2, suppress=True, linewidth=100);

from centroidal_dynamics import *
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
	return np.squeeze(np.asarray(-H))

def skew(x):
    res = zeros([3,3])
    res[0,0] =     0;  res[0,1] = -x[2]; res[0,2] =  x[1];
    res[1,0] =  x[2];  res[1,1] =    0 ; res[1,2] = -x[0];
    res[2,0] = -x[1];  res[2,1] =  x[0]; res[2,2] =   0  ;
    return res
    
    

def __init_6D():
    return zeros([6,3]), zeros(6)
        
def w0(p0, p1, g, p0X, p1X, gX):
    wx, ws = __init_6D()    
    wx[:3,:] = 6*identity(3);  wx[3:,:] = 6*p0X;
    ws[:3]   = 6*(p0 - 2*p1) 
    ws[3:]   = X(-p0, 12*(p1 + g )) 
    return  wx, ws
    
def w1(p0, p1, g, p0X, p1X, gX):
    wx, ws = __init_6D()    
    wx[:3,:] = 3*identity(3);  
    wx[3:,:] = skew(1.5 * (3*p1 - p0))
    ws[:3]   =  1.5 * (3*p0 - 5*p1);
    ws[3:]   = X(3*p0, -p1) + 0.25 * (gX.dot(3*p1 + p0)) 
    return  wx, ws
    
def w2(p0, p1, g, p0X, p1X, gX):
    wx, ws = __init_6D()    
    #~ wx[:3,:] = 0;  
    wx[3:,:] = skew(0.5*g - 3* p0 + 3*p1)
    ws[:3]   =  3*(p0 - p1);
    ws[3:]   = -0.5 * gX.dot(p1) 
    return  wx, ws
    
def w3(p0, p1, g, p0X, p1X, gX):
    wx, ws = __init_6D()    
    wx[:3,:] = -1.5 * identity(3);  
    wx[3:,:] = skew(g - 1.5 * (p1 + p0))
    ws[:3]   = 1.5 * (p1 + p0) 
    #~ ws[3:]   = 0 
    return  wx, ws
    
def w4(p0, p1, g, p0X, p1X, gX):
    wx, ws = __init_6D()    
    wx[:3,:] = -6 *identity(3);  
    wx[3:,:] = skew(g - 6 * p1)
    ws[:3]   = 6*p1 
    #~ ws[3:]   = 0 
    return  wx, ws

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
        self.init_bezier(self._c0, self._dc0, 3)
        
        self._solver = getNewSolver('qpoases', "name")
        #~ self._com_acc_solver  = ComAccLP(self._name+"_comAccLP", self._c0, self._v, self._contact_points, self._contact_normals, 
                                         #~ self._mu, self._g, self._mass, maxIter, verb-1, regularization, solver);
        #~ self._equilibrium_solver = RobustEquilibriumDLP(name+"_robEquiDLP", self._contact_points, self._contact_normals, 
                                                        #~ self._mu, self._g, self._mass, verb=verb-1);
    def init_bezier(c0, dc0, n):
        self._n = n
        self._p0 = c0[:]
        self._p1 = dc0 / n +  self._po  
        self._p0X = skew(self._p0)
        self._p1X = skew(self._p1)
        self.compute_6d_control_point_inequalities()

    def set_contacts(self, contact_points, contact_normals, mu):
        self._contact_points    = np.asarray(contact_points).copy();
        self._contact_normals   = np.asarray(contact_normals).copy();
        self._mu                = mu;
        self._H                 = compute_CWC(self._contact_points, self._contact_normals, mass, mu)#CWC inequality matrix
        #~ self._com_acc_solver.set_contacts(contact_points, contact_normals, mu);
        #~ self._equilibrium_solver = RobustEquilibriumDLP(self._name, contact_points, contact_normals, mu, self._g, self._mass, verb=self._verb);
        
        
    def compute_6d_control_point_inequalities():
        ''' compute the inequality methods that determine the 6D bezier curve w(t)
            as a function of a variable waypoint for the 3D COM trajectory.
            The initial curve is of degree 3 (init pos and velocity, 0 velocity constraints + one free variable).
            The 6d curve is of degree 2*n-2 = 4, thus 5 control points are to be computed. 
            Each control point produces a 6 * 3 inequality matrix wix, and a 6 *1 column right member wsi.
            Premultiplying it by H gives mH w_xi * x <= mH_wsi where m is the mass
            Stacking all of these results in a big inequality matrix A and a column vector x that determines the constraints
            On the 6d curves, Ain x <= Aub
        '''        
            eq = [w0,w1,w2,w3,w4]
            dimH = self._H.shape[0]
            mH = m *self._H 
            self.__Ain =  = zeros(dimH * len(eq),3) 
            self.__Aub = zeros(dimH * len(eq))
            bc = mH.dot(np.concatenate([g,zeros(3)]))  #constant part of Aub, Aubi = bc - mH * wsi
            for i, wi in enumerate(eq):                
                wxi, wsi = wi(self.__p0, self.__p1, self.__g, self.__p0X, self.__p1X, self.__gX)                
                self.__Ain[i*dimH : (i+1)*dimH, : ]  = mH.dot(wxi) #constant part of A, Ac = Ac * wxi
                self.__Aub[i*dimH : (i+1)*dimH    ]  = bc - mH.dot(wsi)
             

    def can_I_stop(self, c0=None, dc0=None, T_0=None, MAX_ITER=None):
        ''' Determine whether the system can come to a stop without changing contacts.
            Keyword arguments:
              c0 -- initial CoM position 
              dc0 -- initial CoM velocity 
              
              T_0 -- an initial guess for the time to stop
            Output: An object containing the following member variables:
              is_stable -- boolean value
              c -- final com position
              dc -- final com velocity. [WARNING] if is_stable is False, not used
              ddc_min -- [WARNING] Not relevant (used)
              t -- [WARNING] always 1 (Bezier curve)
              computation_time -- time taken to solve all the LPs
        '''        
        if T_0 != None or MAX_ITER !=None:
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
        c = ones(3)
        (status, self.x, self.y) = self._solver.solve(self, c, lb= -100 * ones(3), ub = 100 * ones(3), A_in=self.__Ain, Alb=None, Aub=self.__Aub, A_eq=None, b=None)
            
        return Bunch(is_stable=False, c=self._c0+alpha*self._v, dc=Dalpha*self._v, t=t_int, 
                             computation_time=self._computationTime, ddc_min=0.0);

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
    
    
def test(N_CONTACTS = 2, solver='qpoases', verb=0):
    from pinocchio_inv_dyn.multi_contact.utils import generate_contacts, compute_GIWC, find_static_equilibrium_com, can_I_stop
    DO_PLOTS = True;
    PLOT_3D = False;
    mass = 75;             # mass of the robot
    mu = 0.5;           # friction coefficient
    lx = 0.1;           # half foot size in x direction
    ly = 0.07;          # half foot size in y direction
    #First, generate a contact configuration
    CONTACT_POINT_UPPER_BOUNDS = [ 0.5,  0.5,  0.5];
    CONTACT_POINT_LOWER_BOUNDS = [-0.5, -0.5,  0.0];
    gamma = atan(mu);   # half friction cone angle
    RPY_LOWER_BOUNDS = [-2*gamma, -2*gamma, -pi];
    RPY_UPPER_BOUNDS = [+2*gamma, +2*gamma, +pi];
    MIN_CONTACT_DISTANCE = 0.3;
    g_vector = np.array([0., 0., -9.81]);
    X_MARG = 0.07;
    Y_MARG = 0.07;
    
    succeeded = False;
    while(not succeeded):
        (p, N) = generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, 
                                   RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, False);
        X_LB = np.min(p[:,0]-X_MARG);
        X_UB = np.max(p[:,0]+X_MARG);
        Y_LB = np.min(p[:,1]-Y_MARG);
        Y_UB = np.max(p[:,1]+Y_MARG);
        Z_LB = np.min(p[:,2]-0.05);
        Z_UB = np.max(p[:,2]+1.5);
        (H,h) = compute_GIWC(p, N, mu, False);
        (succeeded, c0) = find_static_equilibrium_com(mass, [X_LB, Y_LB, Z_LB], [X_UB, Y_UB, Z_UB], H, h);
        
    dc0 = np.random.uniform(-1, 1, size=3); 
#    dc0[:] = 0;
    
    if(DO_PLOTS):
        f, ax = plut.create_empty_figure();
        for j in range(p.shape[0]):
            ax.scatter(p[j,0], p[j,1], c='k', s=100);
        ax.scatter(c0[0], c0[1], c='r', s=100);
        com_x = np.zeros(2);
        com_y = np.zeros(2);
        com_x[0] = c0[0]; 
        com_y[0] = c0[1];
        com_x[1] = c0[0]+dc0[0]; 
        com_y[1] = c0[1]+dc0[1];
        ax.plot(com_x, com_y, color='b');
        plt.axis([X_LB,X_UB,Y_LB,Y_UB]);           
        plt.title('Contact Points and CoM position'); 
        
    if(PLOT_3D):
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = Axes3D(fig)
#        ax = fig.gca(projection='3d')
        line_styles =["b", "r", "c", "g"];
        ss = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
        ax.scatter(c0[0],c0[1],c0[2], c='k', marker='o');
        for i in range(p.shape[0]):
            ax.scatter(p[i,0],p[i,1],p[i,2], c=line_styles[i%len(line_styles)], marker='o');
            for s in ss:
                ax.scatter(p[i,0]+s*N[i,0],p[i,1]+s*N[i,1],p[i,2]+s*N[i,2], c=line_styles[i%len(line_styles)], marker='x');
        for s in ss:
            ax.scatter(c0[0]+s*dc0[0],c0[1]+s*dc0[1],c0[2]+s*dc0[2], c='k', marker='x');
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z');
   
    if(verb>0):
        print "p:", p.T;
        print "N", N.T;
        print "c", c0.T;
        print "dc", dc0.T;
        
    stabilitySolver = StabilityCriterion("ss", c0, dc0, p, N, mu, g_vector, mass, verb=verb, solver=solver);
    try:
        res = stabilitySolver.can_I_stop();
    except Exception as e:
        print "\n *** New algorithm failed:", e,"\nRe-running the algorithm with high verbosity\n"
        stabilitySolver._verb = 1;
        try:
            res = stabilitySolver.can_I_stop();
        except Exception as e:
            raise
        
    try:
        (has_stopped2, c_final2, dc_final2) = can_I_stop(c0, dc0, p, N, mu, mass, 1.0, 100, verb=verb, DO_PLOTS=DO_PLOTS);
        if((res.is_stable != has_stopped2) or 
            not np.allclose(res.c, c_final2, atol=1e-3) or 
            not np.allclose(res.dc, dc_final2, atol=1e-3)):
            print "\nERROR: the two algorithms gave different results!"
            print "New algorithm:", res.is_stable, res.c, res.dc;
            print "Old algorithm:", has_stopped2, c_final2, dc_final2;
            print "Errors:", norm(res.c-c_final2), norm(res.dc-dc_final2), "\n";
    except Exception as e:
        print "\n\n *** Old algorithm failed: ", e
        print "Results of new algorithm is", res.is_stable, "c0", c0, "dc0", dc0, "cFinal", res.c, "dcFinal", res.dc,"\n";
        
    return (stabilitySolver._computationTime, stabilitySolver._outerIterations, stabilitySolver._innerIterations);
        

#~ if __name__=="__main__":
    #~ N_CONTACTS = 2;
    #~ SOLVER = 'cvxopt'; # cvxopt    
    #~ SOLVER = 'qpoases' # scipy
    #~ VERB = 0;
    #~ N_TESTS = range(0,10);
    #~ time    = np.zeros(len(N_TESTS));
    #~ outIter = np.zeros(len(N_TESTS));
    #~ inIter  = np.zeros(len(N_TESTS));
    #~ # test 392 with 2 contacts give different results
    #~ # test 264 with 3 contacts give different results
    #~ j = 0;
    #~ for i in N_TESTS:
        #~ try:
            #~ np.random.seed(i);
            #~ (time[j], outIter[j], inIter[j]) = test(N_CONTACTS, SOLVER, verb=VERB);
            #~ print "Test %3d, time %3.2f, outIter %3d, inIter %3d" % (i, 1e3*time[j], outIter[j], inIter[j]);
            #~ j += 1;
#~ #            ret = cProfile.run("test()");
        #~ except Exception as e:
            #~ print e;
    #~ print "\nMean computation time %.3f" % (1e3*np.mean(time));
    #~ print "Mean outer iterations %d" % (np.mean(outIter));
    #~ print "Mean inner iterations %d" % (np.mean(inIter));
    #~ print "\nMax computation time %.3f" % (1e3*np.max(time));
    #~ print "Max outer iterations %d" % (np.max(outIter));
    #~ print "Max inner iterations %d" % (np.max(inIter));
