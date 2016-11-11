import numpy as np
from numpy.linalg import norm
from qpoases import PyQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
from pinocchio_inv_dyn.sot_utils import crossMatrix, qpOasesSolverMsg
from pinocchio_inv_dyn.multi_contact.utils import generate_contacts, compute_GIWC, find_static_equilibrium_com, compute_centroidal_cone_generators
import pinocchio_inv_dyn.plot_utils as plut
import matplotlib.pyplot as plt
import time
from math import sqrt, atan, pi
import warnings


EPS = 1e-3; #5
INITIAL_HESSIAN_REGULARIZATION = 1e-8;
MAX_HESSIAN_REGULARIZATION = 1e-4;
FORCE_GENERATOR_COEFFICIENT_UPPER_BOUND = 1e3;
WEIGHT_PARALLEL_CONSTRAINT = 1e3;

class ComAccLP3d (object):
    """
    LP solver dedicated to finding the center of mass (CoM) deceleration that maximizes
    a given linear function, subject to the friction cone constraints.
    The operation amounts to solving the following parametric Linear Program:
      minimize      v' ddc + w sum_i(f_i)
      subject to    P f = A ddc + B c + d
                    f >= 0
    where:
      f         are the contact forces generator coefficients
      ddc       is the 3d CoM acceleration
      c         is the 3d CoM displacement (with respect to its initial position c0)
      w         regularization parameter
    Given a CoM position this class can compute the 
    minimum com acceleration in direction v (i.e. the maximum acceleration in direction -v).
    Since this is a piecewise-linear function of c, it can also compute its derivative with respect 
    to c, and the boundaries of the c-region in which the derivative remains constant.
    """
    
    NO_WARM_START = False;
    
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of constraints (i.e. 6)
    
    Hess = [];     # Hessian
    grad = [];     # gradient
    P = None;       # constraint matrix multiplying the contact force generators
    A = None;       # constraint matrix multiplying the CoM acceleration
    B = None;       # constraint matrix multiplying the CoM position
    d = None;       # constraint vector
    
    mass = 0.0; # robot mass
    g = None;   # 3d gravity vector
    
    maxIter=0;  # max number of iterations
    verb=0;     # verbosity level of the solver (0=min, 2=max)
    
    iter = 0;               # current iteration number
    computationTime = 0.0;  # total computation time
    qpTime = 0.0;           # time taken to solve the QP(s) only
    
    initialized = False;    # true if solver has been initialized
    qpOasesSolver = [];
    options = [];           # qp oases solver's options
    
    epsilon = np.sqrt(np.finfo(float).eps);
    INEQ_VIOLATION_THR = 1e-4;

    def __init__ (self, name, c0, v, contact_points, contact_normals, mu, g, mass, 
                  maxIter=10000, verb=0, grad_reg=1e-5):
        ''' Constructor
            @param c0 Initial CoM position
            @param v Opposite of the direction in which you want to maximize the CoM acceleration (typically that would be
                                                                                                   the CoM velocity direction)
            @param g Gravity vector
            @param regularization Weight of the force minimization, the higher this value, the sparser the solution
        '''
        self.name       = name;
        self.maxIter    = maxIter;
        self.verb       = verb;
        self.m_in       = 6;
        self.initialized    = False;
        self.options        = Options();
        self.options.setToReliable();
        if(self.verb<=1):
            self.options.printLevel  = PrintLevel.NONE;
        elif(self.verb==2):
            self.options.printLevel  = PrintLevel.LOW;
        elif(self.verb==3):
            self.options.printLevel  = PrintLevel.MEDIUM;
        elif(self.verb>3):
            self.options.printLevel  = PrintLevel.DEBUG_ITER;
        self.options.enableRegularisation = False;
        self.options.enableEqualities = True;
#        self.qpOasesSolver.printOptions()
        self.B = np.zeros((6,3));
        self.d = np.empty(6);
        self.c0 = np.empty(3);
        self.v = np.empty(3);
        self.V_ort = np.zeros((2,3));
        self.constrUB = np.zeros(self.m_in)+1e100;
        self.constrLB = np.zeros(self.m_in)-1e100;
        self.set_problem_data(c0, v, contact_points, contact_normals, mu, g, mass, grad_reg);


    def set_com_state(self, c0, v):
        assert np.asarray(c0).squeeze().shape[0]==3, "Com position vector has not size 3"
        assert np.asarray(v).squeeze().shape[0]==3, "Com acceleration direction vector has not size 3"
        self.c0 = np.asarray(c0).squeeze();
        self.v = np.asarray(v).squeeze().copy();
        if(norm(self.v)==0.0):
            raise ValueError("[%s] Norm of com acceleration direction v is zero!"%self.name);
        
        self.v /= norm(self.v);
        self.V_ort[0,:] = np.cross(self.v, np.array([0.0, 0.0, 1.0]));
        if(norm(self.V_ort[0,:])<EPS):
            self.V_ort[0,:] = np.cross(self.v, np.array([0.0, 1.0, 0.0]));
        self.V_ort[0,:] /= norm(self.V_ort[0,:]);
        self.V_ort[1,:] = np.cross(self.v, self.V_ort[0,:]);
        self.Hess[-3:,-3:] = WEIGHT_PARALLEL_CONSTRAINT*np.dot(self.V_ort.T, self.V_ort);

        self.constrMat[:3,-3:] = self.mass*np.identity(3);
        self.constrMat[3:,-3:] = self.mass*crossMatrix(self.c0);
        self.B[3:,:] = -self.mass*crossMatrix(self.g);
        self.d[:3] = self.mass*self.g;
        self.d[3:] = self.mass*np.cross(self.c0, self.g);
        self.grad[-3:] = self.v;
#        self.initialized    = False;


    def set_contacts(self, contact_points, contact_normals, mu, grad_reg=1e-5):
        # compute matrix A, which maps the force generator coefficients into the centroidal wrench
        (self.P, self.G4) = compute_centroidal_cone_generators(contact_points, contact_normals, mu);
        
        # since the size of the problem may have changed we need to recreate the solver and all the problem matrices/vectors
        if(self.n != contact_points.shape[0]*4 + 3):
            self.n              = contact_points.shape[0]*4 + 3;
            self.qpOasesSolver  = SQProblem(self.n,self.m_in); #, HessianType.SEMIDEF);
            self.qpOasesSolver.setOptions(self.options);
            # P f - A ddc = B c + d
            self.constrMat = np.zeros((self.m_in,self.n));
            self.constrMat[:3,-3:] = self.mass*np.identity(3);
            self.constrMat[3:,-3:] = self.mass*crossMatrix(self.c0);
            self.lb = np.zeros(self.n);
            self.lb[-3:] = -1e100;
            self.ub = np.ones(self.n)*FORCE_GENERATOR_COEFFICIENT_UPPER_BOUND;
            self.x  = np.zeros(self.n);
            self.y  = np.zeros(self.n+self.m_in);
            self.Hess = INITIAL_HESSIAN_REGULARIZATION*np.identity(self.n);
            self.Hess[-3:,-3:] = WEIGHT_PARALLEL_CONSTRAINT*np.dot(self.V_ort.T, self.V_ort);
            self.grad = np.ones(self.n);
            self.grad[-3:] = self.v;

        self.grad[:-3] = grad_reg;
        self.constrMat[:,:-3] = self.P;
        self.initialized    = False;


    def set_problem_data(self, c0, v, contact_points, contact_normals, mu, g, mass, grad_reg=1e-5):
        assert np.asarray(g).squeeze().shape[0]==3, "Gravity vector has not size 3"
        assert mass>0.0, "Mass is not positive"
        self.mass = mass;
        self.g = np.asarray(g).squeeze().copy();
        self.set_contacts(contact_points, contact_normals, mu, grad_reg);
        self.set_com_state(c0, v);
        

    def compute_max_deceleration_derivative(self):
        ''' Compute the derivative of the max CoM deceleration (i.e. the solution of the last LP)
            with respect to the parameter alpha (i.e. the CoM position parameter). Moreover, 
            it also computes the bounds within which this derivative is valid (alpha_min, alpha_max).
        '''
        act_set = np.where(self.y[:self.n-3]!=0.0)[0];    # indexes of active bound constraints
        n_as = act_set.shape[0];
        if(n_as > self.n-6):
            raise ValueError("[%s] ERROR Too many active constraints: %d (rather than %d)" % (self.name, n_as, self.n-6));
        if(self.verb>1 and n_as < self.n-6):
            print "[%s] INFO Less active constraints than expected: %d (rather than %d)" % (self.name, n_as, self.n-6);
        self.K = np.zeros((n_as+6,self.n));
        self.k1 = np.zeros((n_as+6,3));
        self.k2 = np.zeros(n_as+6);
        self.K[:n_as,:] = np.identity(self.n)[act_set,:];
        self.K[-6:,:] = self.constrMat;
        self.k1[-6:,:] = self.B;
        act_ub = np.where(abs(self.x[act_set]-self.ub[act_set])<EPS)[0];
        if(self.verb>1 and act_ub.shape[0]>0):
            print "[%s] INFO %d contact forces saturated upper bound"%(self.name, act_ub.shape[0]);
        self.k2[act_ub] = FORCE_GENERATOR_COEFFICIENT_UPPER_BOUND; #self.ub[act_set[act_ub]];
        self.k2[-6:] = self.d;
        U, s, VT = np.linalg.svd(self.K);
        rank = (s > EPS).sum();
        s_inv = (1.0/s[:rank]);
        K_inv_k1 = np.dot(VT[:rank,:].T, np.dot(np.diag(s_inv), np.dot(U[:,:rank].T, self.k1)));
        K_inv_k2 = np.dot(VT[:rank,:].T, s_inv*np.dot(U[:,:rank].T, self.k2));
        if(rank<self.n):
            Z = VT[rank:,:].T;
            P = np.dot(np.dot(Z, np.linalg.inv(np.dot(Z.T, np.dot(self.Hess, Z)))), Z.T);
            K_inv_k1 -= np.dot(P, np.dot(self.Hess, K_inv_k1));
            K_inv_k2 -= np.dot(P, np.dot(self.Hess, K_inv_k2) + self.grad);
            
        # Check that the solution you get by solving the KKT is the same found by the solver
        x_kkt = K_inv_k2; # + K_inv_k1*np.zeros(3)
        if(norm(self.x - x_kkt) > 10*EPS):
            warnings.warn("[%s] ERROR x different from x_kkt. |x-x_kkt|=%f" % (self.name, norm(self.x - x_kkt)));
         
        # store the derivative of the solution w.r.t. the parameter alpha
        dx = K_inv_k1[-3:,:];
        # act_set_mat * delta_c >= act_set_vec
        act_set_mat = K_inv_k1[:-3,:];
        act_set_vec = -K_inv_k2[:-3];
        for i in range(act_set_mat.shape[0]):
            if(norm(act_set_mat[i,:])>EPS):
                act_set_vec[i] /= norm(act_set_mat[i,:]);
                act_set_mat[i,:] /= norm(act_set_mat[i,:]);
        if (np.dot(act_set_mat,np.zeros(3))<act_set_vec-EPS).any():
            raise ValueError("ERROR: after normalization current alpha violates constraints "+str(-act_set_vec));

        return (dx, act_set_mat, act_set_vec);


    def compute_max_deceleration(self, maxIter=None, maxTime=100.0):
        start = time.time();
        if(self.NO_WARM_START):
            self.qpOasesSolver  = SQProblem(self.n,self.m_in);
            self.qpOasesSolver.setOptions(self.options);
            self.initialized = False;            
        if(maxIter==None):
            maxIter = self.maxIter;        
        maxActiveSetIter    = np.array([maxIter]);
        maxComputationTime  = np.array(maxTime);
        #tmp
        self.d[:3] = self.mass*self.g;
        self.d[3:] = self.mass*np.cross(self.c0, self.g);
        # end tmp
        self.constrUB[:6]   = self.d;
        self.constrLB[:6]   = self.constrUB[:6];

        while(True):
            if(not self.initialized):
                self.imode = self.qpOasesSolver.init(self.Hess, self.grad, self.constrMat, self.lb, self.ub, 
                                                     self.constrLB, self.constrUB, maxActiveSetIter, maxComputationTime);
            else:
                self.imode = self.qpOasesSolver.hotstart(self.Hess, self.grad, self.constrMat, self.lb, self.ub, 
                                                         self.constrLB, self.constrUB, maxActiveSetIter, maxComputationTime);
            if(self.imode==0):
                self.initialized = True;
            if(self.imode==0 or 
               self.imode==PyReturnValue.INIT_FAILED_INFEASIBILITY or 
               self.imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY or
               self.Hess[0,0]>=MAX_HESSIAN_REGULARIZATION):
                break;
            self.initialized = False;
            self.Hess *= 10.0;
            maxActiveSetIter    = np.array([maxIter]);
            maxComputationTime  = np.array(maxTime);
            if(self.verb>0):
                print "[%s] WARNING %s. Increasing Hessian regularization to %f"%(self.name, qpOasesSolverMsg(self.imode), self.Hess[0,0]);
            
        self.qpTime = maxComputationTime;
        self.iter   = 1+maxActiveSetIter[0];
        if(self.imode==0):
            self.qpOasesSolver.getPrimalSolution(self.x);
            self.qpOasesSolver.getDualSolution(self.y);

            if((self.x<self.lb-self.INEQ_VIOLATION_THR).any()):
                self.initialized = False;
                raise ValueError("[%s] ERROR lower bound violated" % (self.name)+str(self.x)+str(self.lb));
            if((self.x>self.ub+self.INEQ_VIOLATION_THR).any()):
                self.initialized = False;
                raise ValueError("[%s] ERROR upper bound violated" % (self.name)+str(self.x)+str(self.ub));
            if((np.dot(self.constrMat,self.x)>self.constrUB+self.INEQ_VIOLATION_THR).any()):
                self.initialized = False;
                raise ValueError("[%s] ERROR constraint upper bound violated " % (self.name)+str(np.min(np.dot(self.constrMat,self.x)-self.constrUB)));
            if((np.dot(self.constrMat,self.x)<self.constrLB-self.INEQ_VIOLATION_THR).any()):
                self.initialized = False;
                raise ValueError("[%s] ERROR constraint lower bound violated " % (self.name)+str(np.max(np.dot(self.constrMat,self.x)-self.constrLB)));
        
            (par_ddc_par_c, act_set_mat, act_set_lb) = self.compute_max_deceleration_derivative();
        else:
            self.initialized = False;
            par_ddc_par_c   = np.zeros((3,3));
            act_set_mat     = np.zeros((0,3));
            act_set_lb      = np.zeros(0);
            if(self.verb>0):
                print "[%s] ERROR Qp oases %s" % (self.name, qpOasesSolverMsg(self.imode));
        if(self.qpTime>=maxTime):
            if(self.verb>0):
                print "[%s] Max time reached %f after %d iters" % (self.name, self.qpTime, self.iter);
            self.imode = 9;
        self.computationTime        = time.time()-start;
        return (self.imode, self.x[-3:], par_ddc_par_c, act_set_mat, act_set_lb);


    def getContactForces(self):
        ''' Get the contact forces obtained by solving the last LP '''
        cg = 4;
        nContacts = self.G4.shape[1]/cg;
        f = np.empty((3,nContacts));
        for i in range(nContacts):
            f[:,i] = -np.dot(self.P[:3,cg*i:cg*i+cg], self.x[cg*i:cg*i+cg]);
        return f;
        
    def getContactWrench(self):
        return np.dot(self.P, self.x[:-3]);

    def getGIW(self):
        ddc = self.x[-3:];
        w = np.zeros(6);
        w[:3] = self.mass*(self.g - ddc);
        w[3:] = self.mass*np.cross(self.c0, self.g - ddc);
        return w;

                    
    def reset(self):
        self.initialized = False;
    

        
def test(N_CONTACTS = 2, verb=0):    
    np.set_printoptions(precision=2, suppress=True, linewidth=250);
    DO_PLOTS = False;
    PLOT_3D = False;
    mass = 75;             # mass of the robot
    g_vector = np.array([0,0,-9.81]);
    mu = 0.3;           # friction coefficient
    lx = 0.1;           # half foot size in x direction
    ly = 0.07;          # half foot size in y direction
    USE_DIAGONAL_GENERATORS = True;
    GENERATE_QUASI_FLAT_CONTACTS = False;
    #First, generate a contact configuration
    CONTACT_POINT_UPPER_BOUNDS = [ 0.5,  0.5,  0.5];
    CONTACT_POINT_LOWER_BOUNDS = [-0.5, -0.5,  0.0];
    gamma = atan(mu);   # half friction cone angle
    RPY_LOWER_BOUNDS = [-2*gamma, -2*gamma, -pi];
    RPY_UPPER_BOUNDS = [+2*gamma, +2*gamma, +pi];
    MIN_CONTACT_DISTANCE = 0.3;
    
    lower_margins = np.array([0.07, 0.07, 0.05]);
    upper_margins = np.array([0.07, 0.07, 1.5]);
    
    succeeded = False;
    while(not succeeded):
        (p, N) = generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, 
                                   RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, GENERATE_QUASI_FLAT_CONTACTS);        
        com_lb = np.min(p,0) - lower_margins; 
        com_ub = np.max(p,0) + upper_margins;
        (H,h) = compute_GIWC(p, N, mu, False, USE_DIAGONAL_GENERATORS);
        (succeeded, c0) = find_static_equilibrium_com(mass, com_lb, com_ub, H, h);
        
    dc0 = np.random.uniform(-1, 1, size=3); 
    dc0[2] = 0;
    v = dc0 / norm(dc0);
    
    if(False and DO_PLOTS):
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
        plt.axis([com_lb[0],com_ub[0],com_lb[1],com_ub[1]]);
        plt.title('Contact Points and CoM position'); 
        
    if(PLOT_3D):
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.gca(projection='3d')
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
      
    from utils import compute_com_acceleration_polytope
    from com_acc_LP import ComAccLP
    
    (A, b) = compute_com_acceleration_polytope(c0, H, h, mass, g_vector);
    comAccLP3d = ComAccLP3d("comAccLP3d", c0, dc0, p, N, mu, g_vector, mass, verb=verb, grad_reg=1e-5);
    
    comAccLP   = ComAccLP("comAccLP", c0, dc0, p, N, mu, g_vector, mass, verb=verb, regularization=1e-5);
    i = 0;
    while(i<1):
        (imode, ddc, par_ddc_par_c, act_set_mat, act_set_lb) = comAccLP3d.compute_max_deceleration();
        if(imode!=0):
            print "LP3d failed!"
            
        # compute max deceleration with constraint of being parallel to velocity
#        v = -ddc / norm(ddc);
        v = dc0 / norm(dc0);
        comAccLP.set_com_state(c0, v);
        (imode2, ddAlpha, slope, alpha_min, alpha_max) = comAccLP.compute_max_deceleration(0.0);            
        if(imode2!=0):
            print "LP failed!";
        
        # check whether computed com acceleration is feasible according to GIWC
        error = False;
        acc_ineq = np.dot(A, ddc)-b;
        if(acc_ineq>EPS).any():
            print "\n*** ERROR: acceleration computed by ComAccLP3d is not feasible:", np.max(acc_ineq), "\n";
            error = True;
            
        # check whether deceleration computed by 1d LP is bigger (which should not be possible)
        if(np.dot(v, ddc) < ddAlpha - 1e-3):
            if(norm(ddc)>EPS):
                angle = 180.0*(np.arccos(np.dot(v,ddc)/norm(ddc)) - np.pi)/np.pi;
            else:
                angle = 0.0;
            print "\n*** WARNING: v*ddc<ddAlpha: %.2f %.2f, angle dc-ddc=%.2f deg"%(np.dot(v, ddc), ddAlpha, angle);
        elif(abs(np.dot(v, ddc) - ddAlpha) < EPS):
            pass;
#            print "\n*** GOOD! v*ddc=ddAlpha:", np.dot(v, ddc), ddAlpha, "\n";
        else:
            print "  INFO: ddAlpha=%.2f, v*ddc=%.2f"%(ddAlpha, np.dot(v,ddc));

        # check angle between com acc and com vel
        if(norm(ddc)>EPS):
            angle = 180.0*(np.arccos(np.dot(v,ddc)/norm(ddc)) - np.pi)/np.pi;
            if abs(angle)>1.0:
                print "    angle between dc and ddc: %.2f" % angle, v, ddc/norm(ddc)
            
        if(error):
            f = comAccLP3d.getContactForces();
            f_com = np.sum(f, axis=1);
            ddx_com = (f_com/mass) + g_vector;
            print "Total force at CoM=", f_com;
            print "CoM acc computed from forces=", ddx_com;
            print "Contact wrench =", comAccLP3d.getContactWrench();
            print "GIW            =", comAccLP3d.getGIW();
            print "Contact wrench feasibility:", np.max(np.dot(H, comAccLP3d.getContactWrench()));
            print "GI wrench feasibility:     ", np.max(np.dot(H, comAccLP3d.getGIW()));
            print "Contact forces LP3d\n", comAccLP3d.getContactForces();
            print "Contact forces LP\n", comAccLP.getContactForces();
            print "c", c0.T;
            print "dc", dc0.T;                
            print "LP3d ddc", ddc.T
            print "LP ddAlpha_min = %.3f (slope %.3f, alpha_max %.3f)"%(ddAlpha,slope,alpha_max);
            print "v*ddc", np.dot(v, ddc);
            print "Com acc margin: %.3f"%np.max(acc_ineq);
        
        i += 1;


if __name__=="__main__":
    import cProfile
    N_CONTACTS = 2;
    VERB = 1;
    N_TESTS = range(0,10);
    
    for i in N_TESTS:
        try:
            seed = np.random.randint(0, 1000);
            np.random.seed(seed);
            print "seed %d" % seed
            test(N_CONTACTS, VERB);            
#            ret = cProfile.run("test()");
        except Exception as e:
            print e;
            raise
