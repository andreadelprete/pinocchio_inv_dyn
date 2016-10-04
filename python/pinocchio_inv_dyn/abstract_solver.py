import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.optimize import approx_fprime
from scipy.optimize.slsqp import approx_jacobian
from scipy.optimize import line_search
from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
from qpoases import PySolutionAnalysis as SolutionAnalysis
from qpoases import PyHessianType as HessianType
from qpoases import PySubjectToStatus as SubjectToStatus
from qpoases import PyBooleanType as BooleanType
import time

class AbstractSolver (object):
    """
    Abstract solver class to use as base for all solvers. The basic problem has
    the following structure:
      minimize      0.5*||D*x - d||^2
      subject to    Alb <= A*x <= Aub
                      lb <= x <= ub
      
    NB: Before it was:      
      subject to    G*x + g >= 0 
    where
    """
    
    NO_WARM_START = False;
    
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of inequalities
    
    D = [];     # quadratic cost matrix
    d = [];     # quadratic cost vector
    G = [];     # inequality matrix
    g = [];     # inequality vector
    bounds = []; # bounds on the problem variables
    
    H = [];     # Hessian  H = G^T*G
    dD = [];    # Product dD = d^T*D    
    
    x0 = [];    # initial guess
    solver='';  # type of solver to use
    accuracy=0; # accuracy used by the solver for termination
    maxIter=0;  # max number of iterations
    verb=0;     # verbosity level of the solver (0=min, 2=max)
        
    iter = 0;               # current iteration number
    computationTime = 0.0;  # total computation time
    qpTime = 0.0;           # time taken to solve the QP(s) only
    iterationNumber = 0;    # total number of iterations
    approxProb = 0;         # approximated probability of G*x+g>0
    
    initialized = False;    # true if solver has been initialized
    nActiveInequalities = 0; # number of active inequality constraints
    nViolatedInequalities = 0; # number of violated inequalities
    outerIter = 0;          # number of outer (Newton) iterations
    qpOasesSolver = [];
    options = [];           # qp oases solver's options
    
    softInequalityIndexes = [];
    
    epsilon = np.sqrt(np.finfo(float).eps);
    INEQ_VIOLATION_THR = 1e-4;

    def __init__ (self, n, m_in, solver='slsqp', accuracy=1e-6, maxIter=100, verb=0):
        self.name       = "AbstractSolver";
        self.n          = n;
        self.iter       = 0;
        self.solver     = solver;
        self.accuracy   = accuracy;
        self.maxIter    = maxIter;
        self.verb       = verb;
        self.qpOasesAnalyser= SolutionAnalysis();
        self.changeInequalityNumber(m_in);
        return;
        
    def setSoftInequalityIndexes(self, indexes):
        self.softInequalityIndexes = indexes;
                
    def changeInequalityNumber(self, m_in):
#        print "[%s] Changing number of inequality constraints from %d to %d" % (self.name, self.m_in, m_in);
        self.m_in       = m_in;
        self.iter       = 0;
        self.qpOasesSolver  = SQProblem(self.n,m_in); #, HessianType.SEMIDEF);
        self.options             = Options();
        if(self.verb<=0):
            self.options.printLevel  = PrintLevel.NONE;
        elif(self.verb==1):
            self.options.printLevel  = PrintLevel.LOW;
        elif(self.verb==2):
            self.options.printLevel  = PrintLevel.MEDIUM;
        elif(self.verb>2):            
            self.options.printLevel  = PrintLevel.DEBUG_ITER;
            print "set high print level"
        self.options.enableRegularisation = True;
#        self.options.enableFlippingBounds = BooleanType.FALSE
#        self.options.initialStatusBounds  = SubjectToStatus.INACTIVE
#        self.options.setToMPC();
#        self.qpOasesSolver.printOptions();
        self.qpOasesSolver.setOptions(self.options);
        self.initialized = False;
        
    def setProblemData(self, D, d, A, lbA, ubA, lb, ub, x0=None):
        self.D = D;
        self.d = d.squeeze();
        if(A.shape[0]==self.m_in and A.shape[1]==self.n):
            self.A = A;
            self.lbA = lbA.squeeze();
            self.ubA = ubA.squeeze();
        else:
            print "[%s] ERROR. Wrong size of the constraint matrix, %d rather than %d" % (self.name,A.shape[0],self.m_in);
            
        if(lb.shape[0]==self.n and ub.shape[0]==self.n):
            self.lb = lb.squeeze();
            self.ub = ub.squeeze();
        else:
            print "[%s] ERROR. Wrong size of the bound vectors, %d and %d rather than %d" % (self.name,lb.shape[0], ub.shape[0],self.n);
#        self.bounds = self.n*[(-1e10,1e10)];
        if(x0==None):
            self.x0 = np.zeros(self.n);
        else:
            self.x0 = x0.squeeze();
        
        self.H = np.dot(self.D.T, self.D);
        self.dD = np.dot(self.D.T, self.d);

    def solve(self, D, d, A, lbA, ubA, lb, ub, x0=None, maxIter=None, maxTime=100.0):
        if(self.NO_WARM_START):
            self.qpOasesSolver  = SQProblem(self.n,self.m_in);
            self.qpOasesSolver.setOptions(self.options);
            self.initialized = False;
            
        if(maxIter==None):
            maxIter = self.maxIter;
        self.iter    = 0;
        self.qpTime  = 0.0;
        self.removeSoftInequalities = False;
        self.setProblemData(D,d,A,lbA,ubA,lb,ub,x0);
        start = time.time();
        x = np.zeros(self.x0.shape);
        if(self.solver=='slsqp'):
            (x,fx,self.iterationNumber,imode,smode) = fmin_slsqp(
                                                    self.f_cost, self.x0, 
                                                    fprime=self.f_cost_grad, 
                                                    f_ieqcons=self.f_inequalities, 
                                                    fprime_ieqcons=self.f_inequalities_jac, 
                                                    bounds=self.bounds,
                                                    iprint=self.verb, iter=maxIter,
                                                    acc=self.accuracy,
                                                    full_output = 1);
            self.fx = fx;
            x = np.array(x);
            ''' Exit modes are defined as follows :
                -1 : Gradient evaluation required (g & a)
                 0 : Optimization terminated successfully.
                 1 : Function evaluation required (f & c)
                 2 : More equality constraints than independent variables
                 3 : More than 3*n iterations in LSQ subproblem
                 4 : Inequality constraints incompatible
                 5 : Singular matrix E in LSQ subproblem
                 6 : Singular matrix C in LSQ subproblem
                 7 : Rank-deficient equality constraint subproblem HFTI
                 8 : Positive directional derivative for linesearch
                 9 : Iteration limit exceeded
             '''
            if(self.verb>0 and imode!=0 and imode!=9): #do not print error msg if iteration limit exceeded
                print "[%s] *** ERROR *** %s" % (self.name,smode);
        elif(self.solver=='qpoases'):
#            ubA                 = np.array(self.m_in*[1e9]);
#            lb                  = np.array([ b[0] for b in self.bounds]);
#            ub                  = np.array([ b[1] for b in self.bounds]);
#            A                   = self.get_linear_inequality_matrix();
            self.iter           = 0; #total iters of qpoases
#            lbA                 = -self.get_linear_inequality_vector();
            Hess                = self.f_cost_hess(x);
            grad                = self.f_cost_grad(x);
            self.fx             = self.f_cost(x);
            maxActiveSetIter    = np.array([maxIter - self.iter]);
            maxComputationTime  = np.array(maxTime);
            if(self.initialized==False):
                imode = self.qpOasesSolver.init(Hess, grad, self.A, self.lb, self.ub, self.lbA, self.ubA, maxActiveSetIter, maxComputationTime);
                if(imode==0):
                    self.initialized = True;
            else:
                imode = self.qpOasesSolver.hotstart(Hess, grad, self.A, self.lb, self.ub, self.lbA, self.ubA, maxActiveSetIter, maxComputationTime);
                if(imode==PyReturnValue.HOTSTART_FAILED_AS_QP_NOT_INITIALISED):
                    maxActiveSetIter    = np.array([maxIter]);
                    maxComputationTime  = np.array(maxTime);
                    imode = self.qpOasesSolver.init(Hess, grad, self.A, self.lb, self.ub, self.lbA, self.ubA, maxActiveSetIter, maxComputationTime);
                    if(imode==0):
                        self.initialized = True;

            self.qpTime += maxComputationTime;
            self.iter               = 1+maxActiveSetIter[0];
            self.iterationNumber    = self.iter;  

            ''' if the solution found is unfeasible check whether the initial guess is feasible '''
            self.qpOasesSolver.getPrimalSolution(x);
            ineq_marg = self.f_inequalities(x);
            qpUnfeasible    = False;
            if((ineq_marg<-self.INEQ_VIOLATION_THR).any()):
                qpUnfeasible = True;
                if(x0!=None):
                    ineq_marg       = self.f_inequalities(x0);
                    if(not(ineq_marg<-self.INEQ_VIOLATION_THR).any()):
                        if(self.verb>0):
                            print "[%s] Solution found is unfeasible but initial guess is feasible" % (self.name);
                        qpUnfeasible = False;
                        x = np.copy(x0);
                                    
            ''' if both the solution found and the initial guess are unfeasible remove the soft constraints '''
            if(qpUnfeasible and len(self.softInequalityIndexes)>0):
                # remove soft inequality constraints and try to solve again
                self.removeSoftInequalities = True;
                maxActiveSetIter[0] = maxIter;
                lbAsoft = np.copy(self.lbA);
                ubAsoft = np.copy(self.ubA);
                lbAsoft[self.softInequalityIndexes] = -1e100;
                ubAsoft[self.softInequalityIndexes] = 1e100;
                imode = self.qpOasesSolver.init(Hess, grad, self.A, self.lb, self.ub, lbAsoft, ubAsoft, maxActiveSetIter);
                self.qpOasesSolver.getPrimalSolution(x);
                
                ineq_marg       = self.f_inequalities(x);
                ineq_marg[self.softInequalityIndexes] = 1.0;
                qpUnfeasible    = False;
                if((ineq_marg<-self.INEQ_VIOLATION_THR).any()):
                    ''' if the solution found is unfeasible check whether the initial guess is feasible '''
                    if(x0!=None):
                        x = np.copy(x0);
                        ineq_marg       = self.f_inequalities(x);
                        ineq_marg[self.softInequalityIndexes] = 1.0;
                        if((ineq_marg<-self.INEQ_VIOLATION_THR).any()):
                            print "[%s] WARNING Problem unfeasible even without soft constraints" % (self.name), np.min(ineq_marg), imode;
                            qpUnfeasible = True;
                        elif(self.verb>0):
                            print "[%s] Initial guess is feasible for the relaxed problem" % (self.name);                    
                    else:
                        print "[%s] No way to get a feasible solution (no initial guess)" % (self.name), np.min(ineq_marg);
                elif(self.verb>0):
                    print "[%s] Solution found and initial guess are unfeasible, but relaxed problem is feasible" % (self.name);
            
            if(qpUnfeasible):
                self.print_qp_oases_error_message(imode,self.name);

            if(self.verb>1):
                activeIneq      = np.count_nonzero(np.abs(ineq_marg)<1e-3);
                print "[%s] Iter %d, active inequalities %d" % (self.name,self.iter,activeIneq);            
                    
            # termination conditions
            if(self.iter>=maxIter):
                imode = 9;
                if(self.verb>1):
                    print "[%s] Max number of iterations reached %d" % (self.name, self.iter);
            if(self.qpTime>=maxTime):
                print "[%s] Max time reached %f after %d iters" % (self.name, self.qpTime, self.iter);
                imode = 9;
                    
        elif(self.solver=='sqpoases'):
            ubA = np.array(self.m_in*[1e99]);
            x_newton    = np.zeros(x.shape);
            x           = self.x0;
            A           = self.get_linear_inequality_matrix();
            self.iter   = 0; #total iters of qpoases
            self.outerIter   = 0; # number of outer (Newton) iterations
            while True:
                # compute Newton step
                lb = np.array([ b[0] for b in self.bounds]);
                ub = np.array([ b[1] for b in self.bounds]);
                lb -= x;
                ub -= x;
                lbA = - np.dot(A,x)-self.get_linear_inequality_vector();
#                if(self.outerIter>0):
#                    if((lbA>0.0).any()):
#                        print "[%s] Iter %d lbA[%d]=%f"%(self.name,self.outerIter,np.argmax(lbA),np.max(lbA));
                Hess = self.f_cost_hess(x);
                grad = self.f_cost_grad(x);
                self.fx = self.f_cost(x);
                maxActiveSetIter = np.array([maxIter - self.iter]);
                maxComputationTime  = np.array(maxTime);
                if(self.initialized==False):
                    imode = self.qpOasesSolver.init(Hess, grad, A, lb, ub, lbA, ubA, maxActiveSetIter, maxComputationTime);
                    if(imode==0):
                        self.initialized = True;
                else:
                    imode = self.qpOasesSolver.hotstart(Hess, grad, A, lb, ub, lbA, ubA, maxActiveSetIter, maxComputationTime);
                self.qpTime += maxComputationTime;
                maxTime     -= maxComputationTime;
                # count iterations
                self.iter       += 1+maxActiveSetIter[0];
                self.outerIter  += 1;
                self.qpOasesSolver.getPrimalSolution(x_newton);

                ''' check feasibility of the constraints '''                
                x_new = x + x_newton;
                ineq_marg = self.f_inequalities(x_new);
                if((ineq_marg<-self.INEQ_VIOLATION_THR).any()):
                    if(len(self.softInequalityIndexes)>0 and self.outerIter==1):
                        ''' if constraints are unfeasible at first iteration remove soft inequalities '''
                        if(self.verb>1):
                            print '[%s] Remove soft constraints' % (self.name);
                        self.removeSoftInequalities = True;
                        self.g[self.softInequalityIndexes]   = 1e100;
                        self.iter = 0;
                        continue;
                    elif(len(self.softInequalityIndexes)>0 and self.removeSoftInequalities==False):
                        ''' if constraints are unfeasible at later iteration remove soft inequalities '''
                        if(self.verb>=0):
                            print '[%s] Remove soft constraints at iter %d' % (self.name, self.outerIter);
                        self.removeSoftInequalities = True;                          
                        self.g[self.softInequalityIndexes]   = 1e100;
                        continue;
                    else:
                        if((lbA>0.0).any()):
                            print "[%s] WARNING Problem unfeasible (even without soft constraints) %f" % (self.name,np.max(lbA)), imode;
                        if(self.verb>0):
                            ''' qp failed for some reason (e.g. max iter) '''
                            print "[%s] WARNING imode %d ineq unfeasible iter %d: %f, max(lbA)=%f" % (
                                    self.name, imode, self.outerIter, np.min(ineq_marg), np.max(lbA));
                        break;  ''' if constraints are unfeasible at later iterations exit '''
                
                if(imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY):
                    print "[%s] Outer iter %d QPoases says problem is unfeasible but constraints are satisfied: %f"%(
                            self.name,self.outerIter,np.min(ineq_marg));
                    ind = np.where(ineq_marg<0.0)[0];
                    self.g[ind] -= ineq_marg[ind];
                    continue;
                    
                self.print_qp_oases_error_message(imode,self.name);
                
                # check for convergence
                newton_dec_squared = np.dot(x_newton, np.dot(Hess, x_newton));
                if(0.5*newton_dec_squared < self.accuracy):
                    ineq_marg       = self.f_inequalities(x);
                    if((ineq_marg>=-self.INEQ_VIOLATION_THR).all()):
                        if(self.verb>0):
                            print "[%s] Optimization converged in %d steps" % (self.name, self.iter);
                        break;
                    elif(self.verb>0):
                        v = ineq_marg<-self.INEQ_VIOLATION_THR;
                        print (self.name, self.outerIter, "WARNING Solver converged but inequalities violated:", np.where(v), ineq_marg[v]);
                elif(self.verb>1):
                    print "[%s] Optimization did not converge yet, squared Newton decrement: %f" % (self.name,newton_dec_squared);
                
                # line search
                (alpha, fc, gc, phi, old_fval, derphi) = line_search(self.f_cost, self.f_cost_grad, 
                                                                     x, x_newton, grad, self.fx, 
                                                                     c1=0.0001, c2=0.9);
                x_new = x+alpha*x_newton;
                new_fx = self.f_cost(x_new);
                if(self.verb>1):
                    print "[%s] line search alpha = %f, fc %d, gc %d, old cost %f, new cost %f" % (self.name, alpha, fc, gc, self.fx, new_fx);
                # Check that inequalities are still satisfied
                ineq_marg       = self.f_inequalities(x_new);
                if((ineq_marg<-self.INEQ_VIOLATION_THR).any()):
                    if(self.verb>1):
                        print "[%s] WARNING some inequalities are violated with alpha=%f, gonna perform new line search." % (self.name, alpha);
                    k = 2.0;
                    for i in range(100):
                        alpha = min(k*alpha, 1.0);
                        x_new = x+alpha*x_newton;
                        ineq_marg = self.f_inequalities(x_new);
                        if((ineq_marg>=-self.INEQ_VIOLATION_THR).all()):
                            if(self.verb>1):
                                print "[%s] With alpha=%f the solution satisfies the inequalities." % (self.name, alpha);
                            break;
                        if(alpha==1.0):
                            print "[%s] ERROR With alpha=1 some inequalities are violated, error: %f" % (self.name, np.min(ineq_marg));
                            break;

                x = x_new;
                
                if(self.verb>1):
                    ineq_marg       = self.f_inequalities(x);
                    activeIneq      = np.count_nonzero(np.abs(ineq_marg)<1e-3);
                    nViolIneq       = np.count_nonzero(ineq_marg<-self.INEQ_VIOLATION_THR);
                    print "[%s] Outer iter %d, iter %d, active inequalities %d, violated inequalities %d" % (self.name,self.outerIter,self.iter,activeIneq,nViolIneq);
                
                # termination conditions
                if(self.iter>=maxIter):
                    if(self.verb>1):
                        print "[%s] Max number of iterations reached %d" % (self.name, self.iter);
                    imode = 9;
                    break;
                if(maxTime<0.0):
                    print "[%s] Max time reached %.4f s after %d out iters, %d iters, newtonDec %.6f removeSoftIneq" % (
                        self.name, self.qpTime, self.outerIter, self.iter, newton_dec_squared), self.removeSoftInequalities;
                    imode = 9;
                    break;
                
            self.iterationNumber = self.iter;
        else:
            print '[%s] Solver type not recognized: %s' % (self.name, self.solver);
            return np.zeros(self.n);
        self.computationTime        = time.time()-start;
        ineq = self.f_inequalities(x);
        if(self.removeSoftInequalities):
	        ineq[self.softInequalityIndexes] = 1.0;
        self.nViolatedInequalities  = np.count_nonzero(ineq<-self.INEQ_VIOLATION_THR);
        self.nActiveInequalities    = np.count_nonzero(ineq<1e-3);
        self.imode                  = imode;
        self.print_solution_info(x);
        self.finalize_solution(x);
        return (x, imode);
        
    def finalize_solution(self, x):
        pass;

    def f_cost(self,x):
        e = np.dot(self.D, x) - self.d;
        return 0.5*np.dot(e.T,e);
    
    def f_cost_grad(self,x):
        return approx_fprime(x,self.f_cost,self.epsilon);
        
    def f_cost_hess(self,x):
        return approx_jacobian(x,self.f_cost_grad,self.epsilon);

    def get_linear_inequality_matrix(self):
        return self.A;
          
    def get_linear_inequality_vectors(self):
        return (self.lbA, self.ubA);
        
    def f_inequalities(self,x):
        ineq_marg = np.zeros(2*self.m_in);
        Ax = np.dot(self.get_linear_inequality_matrix(), x);
        ineq_marg[:self.m_in] = Ax - self.lbA;
        ineq_marg[self.m_in:] = self.ubA - Ax;
        return ineq_marg;
          
    def f_inequalities_jac(self,x):
        return self.get_linear_inequality_matrix();
        
    def print_solution_info(self,x):
        if(self.verb>1):        
            print (self.name, "Solution is ", x);
            
    def reset(self):
        self.initialized = False;
        
    def check_grad(self, x=None):
        if(x==None):
            x = np.random.rand(self.n);
        grad = self.f_cost_grad(x);
        grad_fd = approx_fprime(x,self.f_cost,self.epsilon);
        err = np.sqrt(sum((grad-grad_fd)**2));
        print "[%s] Gradient error: %f" % (self.name, err);
        return (grad, grad_fd);
 
    def check_hess(self, x=None):
        if(x==None):
            x = np.random.rand(self.n);
        hess = self.f_cost_hess(x);
        hess_fd = approx_jacobian(x,self.f_cost_grad,self.epsilon);
        err = np.sqrt(np.sum((hess-hess_fd)**2));
        print "[%s] Hessian error: %f" % (self.name, err);
        return (hess, hess_fd);
        
    def print_qp_oases_error_message(self, imode, solver_name):
        if(imode!=0 and imode!=PyReturnValue.MAX_NWSR_REACHED):
            if(imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY):
                print "[%s] ERROR Qp oases HOTSTART_STOPPED_INFEASIBILITY" % solver_name; # 61
            elif(imode==PyReturnValue.MAX_NWSR_REACHED):
                print "[%s] ERROR Qp oases RET_MAX_NWSR_REACHED" % solver_name; # 64
            elif(imode==PyReturnValue.STEPDIRECTION_FAILED_TQ):
                print "[%s] ERROR Qp oases STEPDIRECTION_FAILED_TQ" % solver_name; # 68
            elif(imode==PyReturnValue.STEPDIRECTION_FAILED_CHOLESKY):
                print "[%s] ERROR Qp oases STEPDIRECTION_FAILED_CHOLESKY" % solver_name; # 69
            elif(imode==PyReturnValue.HOTSTART_FAILED_AS_QP_NOT_INITIALISED):
                print "[%s] ERROR Qp oases HOTSTART_FAILED_AS_QP_NOT_INITIALISED" % solver_name; # 54
            elif(imode==PyReturnValue.INIT_FAILED_HOTSTART):
                print "[%s] ERROR Qp oases INIT_FAILED_HOTSTART" % solver_name; # 36
            elif(imode==PyReturnValue.INIT_FAILED_INFEASIBILITY):
                print "[%s] ERROR Qp oases INIT_FAILED_INFEASIBILITY" % solver_name; # 37
            elif(imode==PyReturnValue.UNKNOWN_BUG):
                print "[%s] ERROR Qp oases UNKNOWN_BUG" % solver_name; # 9
            else:
                print "[%s] ERROR Qp oases %d " % (solver_name, imode);
 