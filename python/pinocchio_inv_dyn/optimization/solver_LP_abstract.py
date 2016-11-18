import numpy as np

def enum(**enums):
    return type('Enum', (), enums)

LP_status = enum(OPTIMAL=0,
                 INFEASIBLE=1,
                 UNBOUNDED=2,
                 MAX_ITER_REACHED=3,
                 ERROR=4,
                 UNKNOWN=5
                 );
                 
LP_status_string = ["OPTIMAL", "INFEASIBLE", "UNBOUNDED", "MAX_ITER_REACHED", "ERROR", "UNKNOWN"];

CONSTRAINT_VIOLATION_THR = 1e-5;

class SolverLPAbstract (object):
    """
    Linear Program solver:
         minimize    c' x
         subject to  Alb <= A_in x <= Aub
                     A_eq x = b
                     lb <= x <= ub
    """
    
    def __init__(self, name, maxIter=1000, maxTime=100.0, useWarmStart=True, verb=0):
        self._name = name;
        self._maxIter = maxIter;
        self._maxTime = maxTime;
        self._useWarmStart = useWarmStart;
        self._verb = verb
        self._initialized    = False;
        self._lpTime = 0.0;
        
        
    def reset(self):
        ''' Reset the solver status so that, at the next call of solve(),
            no warm start is used.
        '''
        self._initialized    = False;
        
        
    def set_option(self, key, value):
        ''' Try to set the value of the specified solver option.
            Return True if the operation succeeded, False otherwise.
        '''
        return False;


    def solve(self, c, lb, ub, A_in=None, Alb=None, Aub=None, A_eq=None, b=None):
        ''' Solve the linear program
             minimize    c' x
             subject to  Alb <= A_in x <= Aub
                         A_eq x = b
                         lb <= x <= ub
            Return a tuple containing:
                status flag (LP_status)
                primal solution
                dual solution: Lagrange multipliers in this order: inequalities, bounds, equalities
        '''
        pass;
        
    
    def checkConstraints(self, x, lb, ub, A_in=None, Alb=None, Aub=None, A_eq=None, b=None):
        if((x < lb-CONSTRAINT_VIOLATION_THR).any()):
            self._initialized = False;
            raise ValueError("[%s] ERROR lower bound violated" % (self._name)+str(x)+str(lb));
        if((x > ub+CONSTRAINT_VIOLATION_THR).any()):
            self._initialized = False;
            raise ValueError("[%s] ERROR upper bound violated" % (self._name)+str(x)+str(ub));

        if(A_in is not None):
            assert Aub is not None
            assert Alb is not None
            if((np.dot(A_in,x) > Aub+CONSTRAINT_VIOLATION_THR).any()):
                self._initialized = False;
                raise ValueError("[%s] ERROR constraint upper bound violated " % (self._name)+str(np.min(np.dot(A_in,x)-Aub)));
            if((np.dot(A_in,x) < Alb-CONSTRAINT_VIOLATION_THR).any()):
                self._initialized = False;
                raise ValueError("[%s] ERROR constraint lower bound violated " % (self._name)+str(np.max(np.dot(A_in,x)-Alb)));
        
        if(A_eq is not None):
            if((np.abs(np.dot(A_eq,x)-b) > CONSTRAINT_VIOLATION_THR).any()):
                self._initialized = False;
                raise ValueError("[%s] ERROR equality constraint violated " % (self._name)+str(np.max(np.abs(np.dot(A_eq,x)-b))));

    
    def getUseWarmStart(self):
        ''' Return true if the solver is allowed to warm start, false otherwise.'''
        return self._useWarmStart;
        
    
    def setUseWarmStart(self, useWarmStart): 
        ''' Specify whether the solver is allowed to use warm-start techniques.'''
        self._useWarmStart = useWarmStart;

    
    def getMaximumIterations(self):
        ''' Get the current maximum number of iterations performed by the solver. '''
        return self._maxIter;
        
    
    def setMaximumIterations(self, maxIter):
        ''' Set the current maximum number of iterations performed by the solver. '''
        self._maxIter = maxIter;

    
    def getMaximumTime(self):
        ''' Get the maximum time allowed to solve a problem. '''
        return self._maxTime;
        
    
    def setMaximumTime(self, seconds):
        ''' Set the maximum time allowed to solve a problem. '''
        self._maxIter = seconds;
        
        
    def getLpTime(self):
        ''' Get the time taken by the last LP computation (in seconds). '''
        return self._lpTime;


def getNewSolver(solverType, name, maxIter=1000, maxTime=100.0, useWarmStart=True, verb=0):
    ''' Create a new LP solver of the specified type.
       @param solverType Type of LP solver.
       @return A pointer to the new solver.
    '''
    if(solverType=='qpoases'):
        from solver_LP_qpoases import SolverLPQpOases
        return SolverLPQpOases(name, maxIter, maxTime, useWarmStart, verb);
    if(solverType=='scipy'):
        from solver_LP_scipy import SolverLPScipy
        return SolverLPScipy(name, maxIter, maxTime, useWarmStart, verb);
    if(solverType=='cvxopt'):
        from solver_LP_cvxopt import SolverLPCvxopt
        return SolverLPCvxopt(name, maxIter, maxTime, useWarmStart, verb);

    raise ValueError("[%s] Unrecognized solver type: %s"%(name, solverType));