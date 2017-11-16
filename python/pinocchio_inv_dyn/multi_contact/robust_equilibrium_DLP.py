import numpy as np
#from qpoases import PyQProblem as SQProblem
#from qpoases import PyOptions as Options
#from qpoases import PyPrintLevel as PrintLevel
#from qpoases import PyReturnValue
import time
from utils import compute_centroidal_cone_generators
from pinocchio_inv_dyn.sot_utils import crossMatrix
import pinocchio_inv_dyn.optimization.solver_LP_abstract as optim #import optim.getNewSolver, LP_status, LP_status_string

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
    
#    NO_WARM_START = False;
    
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of inequalities
    
    D = [];     # 
    d = [];     # 
    x = [];     # last solution
    
#    Hess = [];     # Hessian  H = G^T*G
    grad = [];     # gradient
    
    maxIter=0;  # max number of iterations
    verb=0;     # verbosity level of the solver (0=min, 2=max)
    
#    iter = 0;               # current iteration number
    computationTime = 0.0;  # total computation time
#    qpTime = 0.0;           # time taken to solve the QP(s) only
    
#    initialized = False;    # true if solver has been initialized
#    qpOasesSolver = [];
#    options = [];           # qp oases solver's options
    
    INEQ_VIOLATION_THR = 1e-4;

    def __init__ (self, name, contact_points, contact_normals, mu, g, mass, contact_tangents=None, maxIter=100, verb=0, solver='cvxopt'):
        g = np.asarray(g).squeeze();
        self.solver = optim.getNewSolver(solver, name, maxIter=maxIter, verb=verb);
        self.D = np.zeros((6,3));
        self.d = np.zeros(6);
        self.D[3:,:] = -mass*crossMatrix(g);
        self.d[:3]   = mass*g;
        (G, tmp) = compute_centroidal_cone_generators(contact_points, contact_normals, mu, contact_tangents);
        
        self.name = name;
        self.maxIter    = maxIter;
        self.verb       = verb;
        self.m_in       = G.shape[1];
        self.n          = 6;
#        self.iter       = 0;
#        self.initialized = False;
        
#        self.Hess = np.zeros((self.n,self.n));
        self.grad = np.zeros(self.n);
        self.constrInMat = np.zeros((self.m_in,self.n));
        self.constrInMat[:,:] = G.T;
        self.constrInUB = np.zeros(self.m_in) + 1e100;
        self.constrInLB = np.zeros(self.m_in);
        self.constrEqMat = np.dot(np.ones(G.shape[1]), G.T).reshape((1,6));
        self.constrEqB = np.ones(1);
        self.lb = np.array(self.n*[-1e10,]);
        self.ub = np.array(self.n*[1e10,]);
        self.x  = np.zeros(self.n);
        

    def compute_equilibrium_robustness(self, c, maxIter=None, maxTime=100.0):
        start = time.time();
        if(maxIter!=None):
            self.solver.setMaximumIterations(maxIter);
        if(maxTime!=None):
            self.solver.setMaximumTime(maxTime);

        c = np.asarray(c).squeeze();
        self.grad           = np.dot(self.D,c) + self.d;
        (status, self.x, self.y) = self.solver.solve(self.grad, self.lb, self.ub, 
                                                        A_in=self.constrInMat, Alb=self.constrInLB, Aub=self.constrInUB,
                                                        A_eq=self.constrEqMat, b=self.constrEqB);
        self.computationTime        = time.time()-start;
    
        if(status==optim.LP_status.OPTIMAL):
            return (status, np.dot(self.grad, self.x));
        
        return (status, 0.0);
                    
    def reset(self):
        self.initialized = False;
    