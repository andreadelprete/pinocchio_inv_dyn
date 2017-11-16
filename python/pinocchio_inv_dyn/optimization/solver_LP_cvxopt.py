import solver_LP_abstract # import SolverLPAbstract, LP_status
import numpy as np
from cvxopt import matrix, solvers
import time

solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # cvxopt 1.1.7
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions

class SolverLPCvxopt(solver_LP_abstract.SolverLPAbstract):
    """
    Linear Program solver:
      minimize    c' x
     subject to  Alb <= A x <= Aub
                 A_eq x = b
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

        if(n!=self._n or m_in!=self._m_in or m_eq!=self._m_eq):
            self._y = np.zeros(n+m_con);
            self._A_in = np.empty((2*m_in+2*n, n));
            self._A_ub = np.empty(2*m_in+2*n);
            self._n = n;
            self._m_in = m_in;
            self._m_eq = m_eq;
            
        if(m_eq==0):
            A_eq = np.zeros((0,n));
            b = np.zeros(0);
        
        if(m_in>0):
            self._A_in[:m_in,:]             = A_in;
            self._A_in[m_in:2*m_in,:]       = -A_in;    # A x > lb => -A x < -lb
            self._A_ub[:m_in]               = Aub;
            self._A_ub[m_in:2*m_in]         = -Alb;

        self._A_in[2*m_in:2*m_in+n,:]   = np.identity(n);
        self._A_in[2*m_in+n:,:]         = -np.identity(n);
        self._A_ub[2*m_in:2*m_in+n]     = ub;
        self._A_ub[2*m_in+n:]           = -lb;
            
        # Minimize:   c^T * x
        # Subject to: A_ub * x <= b_ub
        #             A_eq * x == b_eq 
        c_cvx = matrix(c);
        A_in_cvx = matrix(self._A_in);
        A_ub_cvx = matrix(self._A_ub);
        A_eq_cvx = matrix(A_eq);
        b_cvx = matrix(b);

#        print "c", c_cvx
#        print "A_in", A_in;
#        print "Aub", A_ub;
#        print "A_eq", A_eq_cvx;
#        print "b", b_cvx

        start = time.time();
        res = solvers.lp(c_cvx, A_in_cvx, A_ub_cvx, A_eq_cvx, b_cvx, solver='glpk');
        self._lpTime    = time.time()-start;

        if(res['status']=='optimal'):
            self._lpStatus = solver_LP_abstract.LP_status.OPTIMAL;
            self._initialized = True;
            self._x = np.array(res['x']).reshape(n);
            self.checkConstraints(self._x, lb, ub, A_in, Alb, Aub, A_eq, b);
            if(res.has_key('iterations')):
                self._iter      = res['iterations'];
            z = np.array(res['z']).reshape(2*m_in+2*n); # Lagrange multipliers for inequality constraints and bounds
#            print "z", z.T
#            print "z!=0", z[z!=0.0].shape
#            print "min z!=0", np.min(np.abs(z[z!=0.0]))
            # take the multiplier corresponding to the active inequality constraints (upper or lower)
            for i in range(m_in):
                if(z[i]!=0.0):
                    self._y[i] = z[i];
                else:
                    self._y[i] = z[i+m_in];
            # take the multiplier corresponding to the active bounds (upper or lower)
            for i in range(n):
                if(z[2*m_in+i]!=0.0):
                    self._y[m_in+i] = z[2*m_in+i];
                else:
                    self._y[m_in+i] = z[2*m_in+n+i];
            if(m_eq>0):
                self._y[-m_eq:] = np.array(res['y']).reshape(m_eq);       # Lagrange multipliers for equality constraints
#            print "Active constraints: ", self._y[self._y!=0.0].shape
        else:
            if(res['status']=='primal infeasible'):
                self._lpStatus = solver_LP_abstract.LP_status.INFEASIBLE;
            elif(res['status']=='dual infeasible'):
                self._lpStatus = solver_LP_abstract.LP_status.UNBOUNDED;
            else:
                self._lpStatus = solver_LP_abstract.LP_status.UNKNOWN;
            self.reset();
            self._x = np.zeros(n);
            self._y = np.zeros(n+m_con);
            if(self._verb>0):
                print "[%s] ERROR cvxopt LP %s" % (self._name, solver_LP_abstract.LP_status_string[self._lpStatus]);
        if(self._lpTime>=self._maxTime):
            if(self._verb>0):
                print "[%s] Max time reached %f after %d iters" % (self._name, self._lpTime, self._iter);
        
        return (self._lpStatus, self._x, self._y);
