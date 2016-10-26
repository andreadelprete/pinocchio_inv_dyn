import numpy as np
from qpoases import PyQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
import time
from utils import compute_centroidal_cone_generators
from pinocchio_inv_dyn.sot_utils import crossMatrix

class RobustEquilibriumDLP (object):
    """
    Solver to compute the robustness measure of the equilibrium of a specified CoM position.
    The operation amounts to solving the following dual LP:
      find          v
      minimize      (d+D c)' v
      subject to    G' v >= 0
                    1' G' v = 1
    where
      -(d+D c)' v   is the robustness measure
      c             is the CoM position
      G             is the matrix whose columns are the gravito-inertial wrench generators
    """
    
    NO_WARM_START = False;
    
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of inequalities
    
    D = [];     # 
    d = [];     # 
    x = [];     # last solution
    
    Hess = [];     # Hessian  H = G^T*G
    grad = [];     # gradient
    
    maxIter=0;  # max number of iterations
    verb=0;     # verbosity level of the solver (0=min, 2=max)
    
    iter = 0;               # current iteration number
    computationTime = 0.0;  # total computation time
    qpTime = 0.0;           # time taken to solve the QP(s) only
    
    initialized = False;    # true if solver has been initialized
    qpOasesSolver = [];
    options = [];           # qp oases solver's options
    
    INEQ_VIOLATION_THR = 1e-4;

    def __init__ (self, name, contact_points, contact_normals, mu, g, mass, maxIter=100, verb=0):
        self.D = np.zeros((6,3));
        self.d = np.zeros(6);
        self.D[3:,:] = -mass*crossMatrix(g);
        self.d[:3]   = mass*g;
        (G, tmp) = compute_centroidal_cone_generators(contact_points, contact_normals, mu);
        
        self.name = name;
        self.maxIter    = maxIter;
        self.verb       = verb;
        self.m_in       = G.shape[1]+1;
        self.n          = 6;
        self.iter       = 0;
        self.qpOasesSolver  = SQProblem(self.n,self.m_in); #, HessianType.SEMIDEF);
        self.options        = Options();
        if(self.verb<=1):
            self.options.printLevel  = PrintLevel.NONE;
        elif(self.verb==2):
            self.options.printLevel  = PrintLevel.LOW;
        elif(self.verb==3):
            self.options.printLevel  = PrintLevel.MEDIUM;
        elif(self.verb>3):
            self.options.printLevel  = PrintLevel.DEBUG_ITER;
            print "set high print level"
        self.options.enableRegularisation = True;
        self.qpOasesSolver.setOptions(self.options);
        self.initialized = False;
        
        self.Hess = np.zeros((self.n,self.n));
        self.grad = np.zeros(self.n);
        self.constrMat = np.zeros((self.m_in,self.n));
        self.constrMat[:-1,:] = G.T;
        self.constrMat[-1,:] = np.dot(np.ones(G.shape[1]), G.T);
        self.constrUB = np.zeros(self.m_in) + 1e100;
        self.constrLB = np.zeros(self.m_in);
        self.constrUB[-1] = 1;
        self.constrLB[-1] = 1;
        self.lb = np.array(self.n*[-1e10,]);
        self.ub = np.array(self.n*[1e10,]);
        self.x  = np.zeros(self.n);
        

    def compute_equilibrium_robustness(self, c, maxIter=None, maxTime=100.0):
        start = time.time();

        if(self.NO_WARM_START):
            self.qpOasesSolver  = SQProblem(self.n,self.m_in);
            self.qpOasesSolver.setOptions(self.options);
            self.initialized = False;
            
        if(maxIter==None):
            maxIter = self.maxIter;
        
        maxActiveSetIter    = np.array([maxIter]);
        maxComputationTime  = np.array(maxTime);
        self.grad           = np.dot(self.D,c) + self.d;
        if(self.initialized==False):
            self.imode = self.qpOasesSolver.init(self.Hess, self.grad, self.constrMat, self.lb, self.ub, self.constrLB, 
                                            self.constrUB, maxActiveSetIter, maxComputationTime);
            if(self.imode!=PyReturnValue.INIT_FAILED_INFEASIBILITY):
                self.initialized = True;
        else:
            self.imode = self.qpOasesSolver.hotstart(self.grad, self.lb, self.ub, self.constrLB, 
                                                     self.constrUB, maxActiveSetIter, maxComputationTime);
            if(self.imode==PyReturnValue.UNKNOWN_BUG):
                self.qpOasesSolver  = SQProblem(self.n,self.m_in);
                self.qpOasesSolver.setOptions(self.options);
                maxActiveSetIter    = np.array([maxIter]);
                maxComputationTime  = np.array(maxTime);
                self.imode = self.qpOasesSolver.init(self.Hess, self.grad, self.constrMat, self.lb, self.ub, self.constrLB, 
                                                     self.constrUB, maxActiveSetIter, maxComputationTime);
        self.qpTime = maxComputationTime;        
        self.iter   = 1+maxActiveSetIter[0];
        self.qpOasesSolver.getPrimalSolution(self.x);
        self.computationTime        = time.time()-start;
        
        if(self.imode!=0 and self.verb>=0):
            self.initialized = False;
            if(self.imode==PyReturnValue.INIT_FAILED_UNBOUNDEDNESS):
                print "[%s] ERROR Qp oases INIT_FAILED_UNBOUNDEDNESS" % self.name; # 38
            elif(self.imode==PyReturnValue.INIT_FAILED_INFEASIBILITY):
                print "[%s] ERROR Qp oases INIT_FAILED_INFEASIBILITY" % self.name; # 37
                return -1;
            elif(self.imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY):
                print "[%s] ERROR Qp oases HOTSTART_STOPPED_INFEASIBILITY" % self.name; # 62
                return -1;
            elif(self.imode==PyReturnValue.HOTSTART_STOPPED_UNBOUNDEDNESS):
                print "[%s] ERROR Qp oases HOTSTART_STOPPED_UNBOUNDEDNESS" % self.name; # 62
            elif(self.imode==PyReturnValue.WORKINGSET_UPDATE_FAILED):
                print "[%s] ERROR Qp oases WORKINGSET_UPDATE_FAILED" % self.name; # 63
            elif(self.imode==PyReturnValue.MAX_NWSR_REACHED):
                print "[%s] ERROR Qp oases MAX_NWSR_REACHED" % self.name; # 64
            elif(self.imode==PyReturnValue.STEPDIRECTION_FAILED_CHOLESKY):
                print "[%s] ERROR Qp oases STEPDIRECTION_FAILED_CHOLESKY" % self.name; # 68
            elif(self.imode==PyReturnValue.HOTSTART_FAILED_AS_QP_NOT_INITIALISED):
                print "[%s] ERROR Qp oases HOTSTART_FAILED_AS_QP_NOT_INITIALISED" % self.name; # 53
#                    RET_INIT_FAILED_HOTSTART = 36
            elif(self.imode==PyReturnValue.UNKNOWN_BUG):
                print "[%s] ERROR Qp oases UNKNOWN_BUG" % self.name; # 9
            else:
                print "[%s] ERROR Qp oases %d " % (self.name, self.imode);
#        else:
#            print "[%s] Problem solved"%(self.name);
        
        if(self.qpTime>=maxTime):
            print "[%s] Max time reached %f after %d iters" % (self.name, self.qpTime, self.iter);
            self.imode = 9;
        
        return np.dot(self.grad, self.x);
                    
    def reset(self):
        self.initialized = False;
    