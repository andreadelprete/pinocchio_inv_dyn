import solver_LP_abstract # import SolverLPAbstract, LP_status
import numpy as np
from scipy.optimize import linprog
import time

class SolverLPScipy (solver_LP_abstract.SolverLPAbstract):
    """
    Linear Program solver:
      minimize    c' x
     subject to  Alb <= A x <= Aub
                 lb <= x <= ub
    """

    def __init__(self, name, maxIter=1000, maxTime=100.0, useWarmStart=True, verb=0):
        solver_LP_abstract.SolverLPAbstract.__init__(self, name, maxIter, maxTime, useWarmStart, verb);
        self._n              = -1;
        self._m_in           = -1;
        
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
        n = c.shape[0];
        m_in = 0 if A_in is None else A_in.shape[0];
        m_eq = 0 if A_eq is None else A_eq.shape[0];
        m_con = m_in+m_eq;

        if(n != self._n or m_in != self._m_in):
            self._y = np.zeros(n+m_in);
            self._A_in = np.empty((m_in*2, n));
            self._A_ub = np.empty(m_in*2);
            self._n = n;
            self._m_in = m_in;
        
        if(m_in>0):
            self._A_in[:m_in,:] = A_in;
            self._A_in[m_in:,:] = -A_in;    # A x > lb => -A x < -lb
            self._A_ub[:m_in] = Aub;
            self._A_ub[m_in:] = -Alb;
        else:
            self._A_in = None;
            self._A_ub = None;
        
#        U, s, VT = np.linalg.svd(A_eq);
#        rank = int((s > 1e-4).sum());
#        x0 = np.dot(VT[:rank,:].T, (1.0/s[:rank])*np.dot(U[:,:rank].T, b));
#        residual = np.dot(A_eq, x0)-b;
#        print "Solution of equality constraints with residual %f:"%(np.linalg.norm(residual))
#        print x0;
#        lb = -ub;
            
        # Minimize:   c^T * x
        # Subject to: A_ub * x <= b_ub
        #             A_eq * x == b_eq 
        start = time.time();
#        print "c", c
#        print "A_in", self._A_in;
#        print "Aub", self._A_ub;
#        print "A_eq", A_eq;
#        print "b", b
#        print "bounds", zip(lb,ub);
        res = linprog(c,  self._A_in, self._A_ub, A_eq, b, zip(lb,ub), 
                      options={"maxiter":self._maxIter, "disp": False});
        self._lpTime    = time.time()-start;
        self._x         = res.x;
        self._iter      = res.nit;
#        self._y         = res.slack;
        
        if(res.status==0):
            self._lpStatus = solver_LP_abstract.LP_status.OPTIMAL;
            self._initialized = True;
            self.checkConstraints(self._x, lb, ub, A_in, Alb, Aub, A_eq, b);
        else:
            if(res.status==1):
                self._lpStatus = solver_LP_abstract.LP_status.MAX_ITER_REACHED;
            elif(res.status==2):
                self._lpStatus = solver_LP_abstract.LP_status.INFEASIBLE;
            elif(res.status==3):
                self._lpStatus = solver_LP_abstract.LP_status.UNBOUNDED;
            else:
                self._lpStatus = solver_LP_abstract.LP_status.UNKNOWN;                
            self.reset();
            self._x = np.zeros(n);
            self._y = np.zeros(n+m_con);
            if(self._verb>0):
                print "[%s] ERROR scipy LP %s, %s" % (self._name, res.message, solver_LP_abstract.LP_status_string[self._lpStatus]);
        if(self._lpTime>=self._maxTime):
            if(self._verb>0):
                print "[%s] Max time reached %f after %d iters" % (self._name, self._lpTime, self._iter);
        
        return (self._lpStatus, self._x, self._y);
