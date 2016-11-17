import solver_LP_abstract # import SolverLPAbstract, LP_status
import numpy as np
#from numpy.linalg import norm
from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
#from qpoases import PySolutionAnalysis as SolutionAnalysis
from pinocchio_inv_dyn.sot_utils import qpOasesSolverMsg
import time

DEFAULT_HESSIAN_REGULARIZATION = 1e-8;

class SolverLPQpOases (solver_LP_abstract.SolverLPAbstract):
    """
    Linear Program solver:
      minimize    c' x
     subject to  Alb <= A x <= Aub
                 lb <= x <= ub
    """

    def __init__(self, name, maxIter=1000, maxTime=100.0, useWarmStart=True, verb=0):
        solver_LP_abstract.SolverLPAbstract.__init__(self, name, maxIter, maxTime, useWarmStart, verb);
        self._hessian_regularization = DEFAULT_HESSIAN_REGULARIZATION;
        self._options        = Options();
        self._options.setToReliable();
        if(self._verb<=1):
            self._options.printLevel  = PrintLevel.NONE;
        elif(self._verb==2):
            self._options.printLevel  = PrintLevel.LOW;
        elif(self._verb==3):
            self._options.printLevel  = PrintLevel.MEDIUM;
        elif(self._verb>3):
            self._options.printLevel  = PrintLevel.DEBUG_ITER;
        self._options.enableRegularisation = False;
        self._options.enableEqualities = True;
        self._n              = -1;
        self._m_con          = -1;
        self._qpOasesSolver  = None;
        
    def set_option(self, key, value):
        if(key=='hessian_regularization'):
            self.set_hessian_regularization(value);
            return True;    
        return False;
        
    def set_hessian_regularization(self, reg):
        assert reg>0, "Hessian regularization must be positive"
        self._hessian_regularization = reg;
        if(self._n>0):
            self._Hess = self._hessian_regularization*np.identity(self._n);

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
    def solve(self, c, lb, ub, A_in=None, Alb=None, Aub=None, A_eq=None, b=None):
        start = time.time();
        n = c.shape[0];
        m_con = 0;
        if((A_in is not None) and (A_eq is not None)):
            m_con = A_in.shape[0] + A_eq.shape[0];
            if(m_con != self._m_con):
                self._A_con = np.empty((m_con,n));
                self._lb_con = np.empty((m_con));
                self._ub_con = np.empty((m_con));
            m_in = A_in.shape[0];
            self._A_con[:m_in,:] = A_in;
            self._lb_con[:m_in] = Alb;
            self._ub_con[:m_in] = Aub;
            self._A_con[m_in:,:] = A_eq;
            self._lb_con[m_in:] = b;
            self._ub_con[m_in:] = b;
        elif(A_in is not None):
            m_con = A_in.shape[0];
            if(m_con != self._m_con):
                self._A_con = np.empty((m_con,n));
                self._lb_con = np.empty((m_con));
                self._ub_con = np.empty((m_con));
            self._A_con[:,:] = A_in;
            self._lb_con[:] = Alb;
            self._ub_con[:] = Aub;
        elif(A_eq is not None):
            m_con = A_eq.shape[0];
            if(m_con != self._m_con):
                self._A_con = np.empty((m_con,n));
                self._lb_con = np.empty((m_con));
                self._ub_con = np.empty((m_con));
            self._A_con[:,:] = A_eq;
            self._lb_con[:] = b;
            self._ub_con[:] = b;
        else:
            m_con = 0;
            if(m_con != self._m_con):
                self._A_con = np.empty((m_con,n));
                self._lb_con = np.empty((m_con));
                self._ub_con = np.empty((m_con));

        if(n != self._n or m_con != self._m_con):
            self._qpOasesSolver = SQProblem(n, m_con); #, HessianType.SEMIDEF);
            self._qpOasesSolver.setOptions(self._options);
            self._Hess = self._hessian_regularization*np.identity(n);
            self._x = np.zeros(n);
            self._y = np.zeros(n+m_con);
            self._n = n;
            self._m_con = m_con;
            
        maxActiveSetIter    = np.array([self._maxIter]);
        maxComputationTime  = np.array(self._maxTime);

        if(not self._initialized):
            self._imode = self._qpOasesSolver.init(self._Hess, c, self._A_con, lb, ub, self._lb_con, 
                                                   self._ub_con, maxActiveSetIter, maxComputationTime);
        else:
            self._imode = self._qpOasesSolver.hotstart(self._Hess, c, self._A_con, lb, ub, self._lb_con, 
                                                       self._ub_con, maxActiveSetIter, maxComputationTime);
            
        self._lpTime = maxComputationTime;
        self._iter   = 1+maxActiveSetIter[0];
        if(self._imode==0):
            self._initialized = True;
            self._lpStatus = solver_LP_abstract.LP_status.OPTIMAL;
            self._qpOasesSolver.getPrimalSolution(self._x);
            self._qpOasesSolver.getDualSolution(self._y);
            self.checkConstraints(self._x, lb, ub, A_in, Alb, Aub, A_eq, b);
        else:
            if(self._imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY or
               self._imode==PyReturnValue.INIT_FAILED_INFEASIBILITY):
                self._lpStatus = solver_LP_abstract.LP_status.INFEASIBLE;
            elif(self._imode==PyReturnValue.MAX_NWSR_REACHED):
                self._lpStatus = solver_LP_abstract.LP_status.MAX_ITER_REACHED;
            else:
                self._lpStatus = solver_LP_abstract.LP_status.ERROR;
                
            self.reset();
            self._x = np.zeros(n);
            self._y = np.zeros(n+m_con);
            if(self._verb>0):
                print "[%s] ERROR Qp oases %s" % (self._name, qpOasesSolverMsg(self._imode));
        if(self._lpTime>=self._maxTime):
            if(self._verb>0):
                print "[%s] Max time reached %f after %d iters" % (self._name, self._lpTime, self._iter);
            self._imode = 9;
        self._computationTime        = time.time()-start;
        return (self._lpStatus, self._x, self._y);
