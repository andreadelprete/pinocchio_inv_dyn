# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:54:39 2016

@author: stonneau
"""

from pinocchio_inv_dyn.optimization.solver_LP_abstract import LP_status, LP_status_string
#~ from pinocchio_inv_dyn.sot_utils import qpOasesSolverMsg
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

__EPS = 1e-5;

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

def compute_w(c, ddc, dL=array([0.,0.,0.]), m = 54., g_vec=array([0.,0.,-9.81])):
    w1 = m * (ddc - g_vec)
    return array(w1.tolist() + (X(c, w1) + dL).tolist())

def is_stable(H,c=array([0.,0.,0.]), ddc=array([0.,0.,0.]), dL=array([0.,0.,0.]), m = 54., g_vec=array([0.,0.,-9.81]), robustness = 0.):
    w = compute_w(c, ddc, dL, m, g_vec) 
    print "rob,", np.max(H.dot(w))
    return (H.dot(w)<=-robustness).all()

def skew(x):
    res = zeros([3,3])
    res[0,0] =     0;  res[0,1] = -x[2]; res[0,2] =  x[1];
    res[1,0] =  x[2];  res[1,1] =    0 ; res[1,2] = -x[0];
    res[2,0] = -x[1];  res[2,1] =  x[0]; res[2,2] =   0  ;
    return res
    
    

def __init_6D():
    return zeros([6,3]), zeros(6)
      
def normalize(A,b):
    for i in range (A.shape[0]):
        n_A = norm(A[i,:])
        if(n_A != 0.):
            A[i,:] = A[i,:] / n_A
            b[i] = b[i] / n_A
    return A, b
       
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

def __check_trajectory(p0,p1,p2,p3,T,H, mass, g, time_step = 0.1):
    robustness = -0.000001
    if(time_step < 0):
        time_step = 0.01
        robustness = -0.00000001
    resolution = int(T / time_step)
    wps = [p0,p1,p2,p3]; wps = matrix([pi.tolist() for pi in wps]).transpose()
    c_t = bezier(wps)
    ddc_t = c_t.compute_derivate(2)
    def c_tT(t):
        return asarray(c_t(t/T)).flatten()
    def ddc_tT(t):
        return 1./(T*T) * asarray(ddc_t(t/T)).flatten()
    for i in range(resolution):
        t = T * float(i) / float(resolution)
        if not (is_stable(H,c=c_tT(t), ddc=ddc_tT(t), dL=array([0.,0.,0.]), m = mass, g_vec=g, robustness = -0.00001)):
            raise ValueError("trajectory is not stale !")
        
        

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
        self._solver = getNewSolver('qpoases', "name")
        
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
        
        
    #~ def __compute_wixs(self, T, num_step = -1):        
        #~ alpha = 1 / (T*T)
        #~ wps = [wi(self._p0, self._p1, self._g, self._p0X, self._p1X, self._gX, alpha)  for wi in wis]
        #~ if num_step > 0:
            #~ dt = float(T/num_step)
            #~ wps_bern = [ [ (b(i*dt)*wps[idx][0], b(i*dt)*wps[idx][1]) for idx,b in enumerate(b4)] for i in range(num_step + 1) ]
            #~ wps = [reduce(lambda a, b : (a[0] + b[0], a[1] + b[1]), wps_bern_i) for wps_bern_i in wps_eval]
        #~ return wps
        
        
    def __compute_wixs(self, T, num_step = -1):        
        alpha = 1 / (T*T)
        wps = [wi(self._p0, self._p1, self._g, self._p0X, self._p1X, self._gX, alpha)  for wi in wis]
        if num_step > 0:
            dt = (float(T)/num_step)
            res = []
            for i in range(num_step+1):
                (wxres, wsres) = (zeros([6,3]),zeros(6))
                for j in range(0,5):
                    (wxi, wsi) = wps[j]
                    bi = b4[j](i*dt)
                    wxres += wxi * bi
                    wsres += wsi * bi
                res +=[(wxres, wsres)]
            return res
            #~ wps_bern = [ [ (b(i*dt)*wps[idx][0], b(i*dt)*wps[idx][1]) for idx,b in enumerate(b4)] for i in range(num_step + 1) ]
            #~ wps = [reduce(lambda a, b : (a[0] + b[0], a[1] + b[1]), wps_bern_i) for wps_bern_i in wps_eval]
        return wps
        
    def __add_kinematic_and_normalize(self,A,b):        
        if self._kinematic_constraints != None:
            dim_kin = self._kinematic_constraints[0].shape[0]
            A[-dim_kin:,:] = self._kinematic_constraints[0][:]
            b[-dim_kin:] =  self._kinematic_constraints[1][:]
        A, b = normalize(A,b)
        self.__Ain = A[:]; self.__Aub = b[:]
        
    def _compute_num_steps(self, T, time_step):    
        num_steps = -1    
        if(time_step > 0):
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
        c = zeros(3);c[2] = -1
        self.compute_6d_control_point_inequalities(T, time_step)
        (status, x, y) = self._solver.solve(c, lb= -100 * ones(3), ub = 100 * ones(3), A_in=self.__Ain, Alb=-100000* ones(self.__Ain.shape[0]), Aub=self.__Aub, A_eq=None, b=None)
        
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
    
from pinocchio_inv_dyn.multi_contact.utils import generate_contacts, compute_GIWC, find_static_equilibrium_com, can_I_stop, check_static_eq
def test(N_CONTACTS = 2, solver='qpoases', verb=0):
    
    DO_PLOTS = False;
    PLOT_3D = False;
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
    global mass
    global g_vector
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
        Z_LB = np.max(p[:,2]+0.3);
        Z_UB = np.max(p[:,2]+1.5);
        (H,h) = compute_GIWC(p, N, mu, False);     
        (succeeded, c0) = find_static_equilibrium_com(mass, [X_LB, Y_LB, Z_LB], [X_UB, Y_UB, Z_UB], H, h);
        
    dc0 = np.random.uniform(-1, 1, size=3); 
    
    Z_MIN = np.max(p[:,2])-0.1;
    Ineq_kin = zeros([3,3]); Ineq_kin[2,2] = -1
    ineq_kin = zeros(3); ineq_kin[2] = -Z_MIN
    
    
    bezierSolver = BezierZeroStepCapturability("ss", c0, dc0, p, N, mu, g_vector, mass, verb=verb, solver=solver, kinematic_constraints = [Ineq_kin,ineq_kin ]);
    #~ bezierSolver = BezierZeroStepCapturability("ss", c0, dc0, p, N, mu, g_vector, mass, verb=verb, solver=solver);
    stabilitySolver = StabilityCriterion("ss", c0, dc0, p, N, mu, g_vector, mass, verb=verb, solver=solver);
    window_times = [1]+ [0.1*i for i in range(1,22)] #try nominal time first
    #~ window_times = [1]
    found = False
    time_step_check = 0.1
    for i, el in enumerate(window_times):
        if (found):
            break
        res = bezierSolver.can_I_stop(T=el, time_step = time_step_check);
        #~ res = bezierSolver.can_I_stop(T=el, time_step = -1);
        if (res.is_stable):
            found = True
            print "found at ", el
            __check_trajectory(bezierSolver._p0, bezierSolver._p1, res.c, res.c, el, bezierSolver._H, 
                               bezierSolver._mass, bezierSolver._g, time_step = time_step_check)
            if i != 0:
                print "Failed to stop at 1, but managed to stop at ", el
    #~ except ValueError as e:
        #~ print e
    #~ except Exception as e:
        #~ print "\n *** New algorithm failed:", e,"\nRe-running the algorithm with high verbosity\n"
    
    #~ print "out"
    
    try:
        res2 = stabilitySolver.can_I_stop();
    except Exception as e:
        pass
        #~ print "\n\n *** Old algorithm failed: ", e
        #~ print "Results of new algorithm is", res.is_stable, "c0", c0, "dc0", dc0, "cFinal", res.c, "dcFinal", res.dc,"\n";
        
    
    return res.is_stable, res2.is_stable, res, res2, c0, dc0, H, h, p, N
        

if __name__=="__main__":        
    g_vector = np.array([0., 0., -9.81]);
    mass = 75;             # mass of the robot
    from pinocchio_inv_dyn.multi_contact.stability_criterion import  StabilityCriterion
    from matplotlib import rcParams
    rcParams.update({'font.size': 11})
    mine_won = 0
    mine_lose = 0
    total_stop = 0
    total_not_stop = 0
    total_disagree = 0
    margin_i_win_he_lose = [] # remaining speed
    margin_he_wins_i_lost = [] # remaining acceleration
    curves_when_i_win = []
    #~ times_disagree = []
    #~ times_agree_stop = []
    
    num_tested = 0.
    for i in range(100):
        num_tested = i-1
        mine, theirs, r_mine, r_theirs, c0, dc0, H,h, p, N = test()
        #~ print "H test", H.shape 
        if(mine != theirs):
            total_disagree+=1
            if(mine):
                #~ times_disagree +=[r_mine.t]
                margin_i_win_he_lose+=[r_theirs.dc]
                curves_when_i_win+=[(c0[:], dc0[:], r_theirs.c[:], r_theirs.dc[:], r_mine.t, r_mine.c_of_t, r_mine.dc_of_t, r_mine.ddc_of_t, H[:], h[:], p[:], N)]
                print "margin when he lost: ", norm(r_theirs.dc)
            #~ else:
                #~ times_disagree +=[r_theirs.t]
            if mine:
                mine_won +=1
            else:
                mine_lose +=1
        elif(mine or theirs):
            total_stop+=1
            #~ times_agree_stop+=[r_mine.t]
            margin_he_wins_i_lost+=[r_theirs.ddc_min]
            
            #~ margin_i_win_he_lose+=[r_theirs.dc]
            #~ curves_when_i_win+=[(c0[:], dc0[:], r_theirs.c[:], r_theirs.dc[:], r_mine.t, r_mine.c_of_t, r_mine.dc_of_t, r_mine.ddc_of_t, H[:], h[:], p[:], N)]
            #~ print "margin when he wins: ", r_theirs.ddc_min
        else:
            total_not_stop+=1
    
    print "% of stops", 100. * float(total_stop) / num_tested, total_stop
    print "% of total_disagree", 100. * float(total_disagree) / num_tested, total_disagree
    if total_disagree > 0:
        print "% of wins", 100. * float(mine_won) / total_disagree
        print "% of lose", 100. * float(mine_lose) / total_disagree
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
            
    def __plot_3d_points(ax, points, c = 'b'):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        zs = [point[2] for point in points]
        ax.scatter(xs[:1], ys[:1], zs[:1], c='r')       
        ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1], c=c)       
        ax.scatter(xs[-1:], ys[-1:], zs[-1:], c='g')        
        ax.set_xlabel('X Label', fontsize = 11)
        ax.set_ylabel('Y Label', fontsize = 11)
        ax.set_zlabel('Z Label', fontsize = 11)
        
    from pinocchio_inv_dyn.geom_utils import is_vector_inside_cone, plot_inequalities
    def plot_support_polygon(H,h,p,N,ax, c0):
        from pinocchio_inv_dyn.multi_contact.utils import generate_contacts, find_static_equilibrium_com, compute_GIWC, compute_support_polygon
        #~ (H,h) = compute_GIWC(p, N, mu);        
        global mass
        global g_vector
        (B_sp, b_sp) = compute_support_polygon(H, h, mass, g_vector, eliminate_redundancies=False);
        X_MIN = np.min(p[:,0]);
        X_MAX = np.max(p[:,0]);
        X_MIN -= 0.5*(X_MAX-X_MIN);
        X_MAX += 0.5*(X_MAX-X_MIN);
        Y_MIN = np.min(p[:,1]);
        Y_MAX = np.max(p[:,1]);
        Y_MIN -= 0.5*(Y_MAX-Y_MIN);
        Y_MAX += 0.5*(Y_MAX-Y_MIN);
        num_steps = 50
        dx = (X_MAX - X_MIN) / float(num_steps)
        dy = (Y_MAX - Y_MIN) / float(num_steps)
        #~ points = [(X_MIN + dx * i, Y_MAX + dy * j, 0.) for i in range(num_steps+1) for j in range(num_steps+1) if check_static_eq(H, h, mass, array([X_MIN + dx * i, Y_MAX + dy * j,0.]), g_vector) ]
        #~ points = [c0]+[[X_MIN + dx * i, Y_MIN + dy * j, -0.5] for i in range(num_steps+1) for j in range(num_steps+1) if check_static_eq(H, h, mass, [X_MIN + dx * i, Y_MAX + dy * j,0.], g_vector)]
        points = [c0]+[[X_MIN + dx * i, Y_MIN + dy * j, 0] for i in range(num_steps+1) for j in range(num_steps+1) ]
        pts2  = []
        for pt in points:
            if check_static_eq(H, h, mass, pt, g_vector):
                pts2 += [pt]
        __plot_3d_points(ax, pts2, c="r")   
        #~ __plot_3d_points(ax, points2, c="r")   
        __plot_3d_points(ax, p, c="r")   
        #~ for i in range(num_steps):
            #~ for j in range(num_steps):
        #~ plot_inequalities(B_sp, b_sp, [X_MIN,X_MAX], [Y_MIN,Y_MAX], ax=ax, color='b', lw=4, is_3d=False);
        #~ plot_inequalities(B_sp, b_sp, [X_MIN,X_MAX], [Y_MIN,Y_MAX], ax=ax, color='b', lw=4, is_3d=False);
        #~ plt.show();
    
    def plot_win_curve(n = -1, num_pts = 20):
        global curves_when_i_win
        if n > len(curves_when_i_win) -1 or n < 0:
            print "n bigger than num curves or equal to -1, plotting last curve"
            n = len(curves_when_i_win) -1        
        c0, dc0, c_end, dc_end, t_max, c_of_t, dc_of_t, ddc_of_t, H, h, p, N = curves_when_i_win[n]
        print "c0 ", c0
        print "Is c0 stable ? ", check_static_eq(H, h, mass, c0, g_vector)
        print "Is end stable ? ", check_static_eq(H, h, mass, c_of_t(t_max), g_vector)
        
        w = np.zeros(6);
        w[2] = -mass*9.81;
        w[3:] = mass*np.cross(c_of_t(t_max), g_vector);
        print 'max ', np.max(np.dot(H, w) - h)
        
        X_MIN = np.min(p[:,0]);
        X_MAX = np.max(p[:,0]);
        X_MIN -= 0.1*(X_MAX-X_MIN);
        X_MAX += 0.1*(X_MAX-X_MIN);
        Y_MIN = np.min(p[:,1]);
        Y_MAX = np.max(p[:,1]);
        print "Is XMIN ? ", X_MIN
        print "Is XMAX ? ", X_MAX
        print "Is YMIN ? ", Y_MIN
        print "Is YMAX ? ", Y_MAX
        delta = t_max / float(num_pts)
        fig = plt.figure()  
        ax = fig.add_subplot(221, projection='3d')
        #~ ax = fig.add_subplot(221)
        __plot_3d_points(ax, [c_of_t(i * delta) for i in range(num_pts)])
        __plot_3d_points(ax, [c0 + (c_end-c0)* i * delta for i in range(num_pts)], c = "y")
        plot_support_polygon(H,h,p,N,ax, c0)
        ax = fig.add_subplot(222, projection='3d')
        __plot_3d_points(ax, [dc_of_t(i * delta) for i in range(num_pts)])    
        __plot_3d_points(ax, [dc0 + (dc_end-dc0)* i * delta for i in range(num_pts)], c = "y")    
        ax = fig.add_subplot(223, projection='3d')
        __plot_3d_points(ax, [ddc_of_t(i * delta) for i in range(num_pts)])
        #~ ax = fig.add_subplot(224, projection='3d')
        __plot_3d_points(ax, [-dc0* i * delta for i in range(num_pts)], c = "y")
        #~ ax = fig.add_subplot(121, projection='3d')
        #~ __plot_3d_points(ax, [ddc_of_t(i * delta) for i in range(num_pts)])
        #~ ax = fig.add_subplot(122, projection='3d')
        #~ __plot_3d_points(ax, [-dc0* i * delta for i in range(num_pts)])
        #~ print "cross product ", X(-dc0,ddc_of_t(0.5) - ddc_of_t(0) ) / norm(X(-dc0,ddc_of_t(0.5) - ddc_of_t(0) ))
        #~ print "init acceleration ", ddc_of_t(0)
        print "init velocity ", dc_of_t(0)
        print "end velocity ", dc_of_t(t_max)
        #~ print "cross product ", X(-dc0,ddc_of_t(t_max) - ddc_of_t(0) ) / norm(X(-dc0,ddc_of_t(t_max) - ddc_of_t(0) ))
        
        #~ plt.show()
        
    def plot_n_win_curves(n = -1, num_pts = 50):
        global curves_when_i_win
        if n > len(curves_when_i_win) -1 or n < 0:
            print "n bigger than num curves or equal to -1, plotting last curve"
            n = len(curves_when_i_win) -1
        for i in range(n):
            plot_win_curve(i, num_pts)
        plt.show()
        
