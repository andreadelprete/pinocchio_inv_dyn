from abstract_solver import AbstractSolver
import numpy as np

class StandardQpSolver (AbstractSolver):
    """
    Nonrobust inverse dynamics solver for the problem:
    min ||D*x - d||^2
    s.t.  lbA <= A*x <= ubA 
    """

    def __init__(self, n, m_in, solver='slsqp', accuracy=1e-6, maxIter=100, verb=0):
        AbstractSolver.__init__(self, n, m_in, solver, accuracy, maxIter, verb);
        self.name = "Classic TSID";
        
    def f_cost(self,x):
        e = np.dot(self.D, x) - self.d;
        return 0.5*np.dot(e.T,e);
    
    def f_cost_grad(self,x):
        return np.dot(self.H,x) - self.dD;
        
    def f_cost_hess(self,x):
        return self.H;

    def get_linear_inequality_matrix(self):
        return self.A;
          
    def get_linear_inequality_vectors(self):
        return (self.lbA, self.ubA);
