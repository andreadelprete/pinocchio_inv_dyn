import numpy as np
from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
from math import sqrt
import time

class StaggeredProjections (object):
    """
    Algorithm described in "Staggered projections for frictional contact in multibody systems".
    """
    USE_DIAGONAL_GENERATORS = True;
    
    name = "";      # name of this instance
    n = 0;          # number of variables
    k = 0;          # number of contact points
    
    D = [];         # friction cost matrix
    ET = [];        # friction inequality matrix
    mu_alpha = [];  # friction inequality upper-bound vector
    N = [];         # contact cost matrix
    
    bounds_contact = []; # bounds on the problem variables
    bounds_friction = []; # bounds on the problem variables
    
    H_contact = [];      # Hessian  H = G^T*G
    g_contact = [];      # Product dD = d^T*D    
    H_friction = [];     # Hessian  H = G^T*G
    g_friction = [];     # Product dD = d^T*D    
    
    alpha = []      # contact impulses
    beta = [];      # friciton impulses
    f0 = [];        # initial guess for the friction impulses projected in joint space
    
    accuracy=0; # accuracy used by the solver for termination
    maxIter=0;  # max number of iterations
    verb=0;     # verbosity level of the solver (0=min, 2=max)
        
    iter = 0;               # current iteration number
    iter_contact = 0;
    iter_friction = 0;
    
    qpTime_contact = 0.0;
    qpTime_friction = 0.0;
    
    t                      = 0;     # number of times the method simulate has been called
    meanComputationTime    = 0.0;
    maxComputationTime     = 0.0;
    maxIterations          = 0;
    meanIterations         = 0.0;
        
    initialized_contact = False;    # true if solver has been initialized
    initialized_friction = False;    # true if solver has been initialized
    
    solverContact = [];
    solverFriction = [];
    options = [];           # qp oases solver's options
    
    epsilon = np.sqrt(np.finfo(float).eps);
    INEQ_VIOLATION_THR = 1e-4;

    def __init__ (self, n, mu, accuracy=1e-4, maxIter=30, verb=0):
        self.name       = "StagProj";
        self.n          = n;
        self.mu         = mu;
        self.iter       = 0;
        self.accuracy   = accuracy;
        self.maxIter    = maxIter;
        self.verb       = verb;
        self.options            = Options();
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
        self.changeContactsNumber(0);
        self.S_T = np.zeros((self.n+6,self.n));
        self.S_T[6:6+self.n,:] = np.identity(self.n);
        self.t                      = 0;
        self.meanComputationTime    = 0.0;
        self.maxComputationTime     = 0.0;
        self.meanIterations         = 0.0;
        self.maxIterations          = 0.0;
        return;
        
    def changeContactsNumber(self, k):
        if(self.verb>1):
            print "[%s] Changing number of contacts from %d to %d" % (self.name, self.k, k);
        self.k          = k;
        self.iter       = 0;
        self.initialized_contact = False;
        self.initialized_friction = False;
        if(k>0):
            self.solverContact      = SQProblem(self.k, 0);
            self.solverFriction     = SQProblem(4*self.k, self.k);
            self.solverContact.setOptions(self.options);
            self.solverFriction.setOptions(self.options);
            
        self.contact_forces     = np.zeros(self.k*3);
        self.f0                 = np.zeros(self.n+6); # reset initial guess
        self.bounds_contact     = self.k*[(0.0,1e9)];
        self.bounds_friction    = (4*self.k)*[(0.0,1e9)];
        self.beta               = np.zeros(4*self.k);

        self.N = np.zeros((self.n+6, self.k));
        self.dJv = np.zeros(self.k);
        self.lb_contact     = np.array([ b[0] for b in self.bounds_contact]);
        self.ub_contact     = np.array([ b[1] for b in self.bounds_contact]);
        self.alpha          = np.zeros(self.k);
        
        ''' compute constraint matrix '''
        self.ET = np.zeros((self.k, 4*self.k));
        for i in range(0,self.k):
            self.ET[i,4*i:4*i+4] = 1;
        
        self.D = np.zeros((self.n+6, 4*self.k));
        
        ''' compute tangent directions matrix '''
        self.T = np.zeros((3, 4)); # tangent directions
        if(self.USE_DIAGONAL_GENERATORS):
            tmp = 1.0/sqrt(2.0);
            self.T[:,0] = np.array([+tmp, +tmp, 0]);
            self.T[:,1] = np.array([+tmp, -tmp, 0]);
            self.T[:,2] = np.array([-tmp, +tmp, 0]);
            self.T[:,3] = np.array([-tmp, -tmp, 0]);
        else:
            self.T[:,0] = np.array([+1,  0, 0]);
            self.T[:,1] = np.array([-1,  0, 0]);
            self.T[:,2] = np.array([ 0, +1, 0]);
            self.T[:,3] = np.array([ 0, -1, 0]);
        
        self.lb_friction     = np.array([ b[0] for b in self.bounds_friction]);
        self.ub_friction     = np.array([ b[1] for b in self.bounds_friction]);
        self.lbA_friction    = -1e100*np.ones(self.k);
        
    def simulate(self, v, M, h, tau, dt, contactList, dJv_list, maxIter=None, maxTime=100.0):
        start = time.time();

        if(maxIter==None):
            maxIter = self.maxIter;
        self.maxTime = maxTime;
        self.iter    = 0;
        self.min_res = 1e10; # minimum residual
        self.rel_err = 1e10; # relative error
        self.contactList = contactList;
        self.dJv_list = dJv_list;
        self.dt = dt;
        
        ''' If necessary resize the solvers '''
        if(self.k!=len(contactList)):
            self.changeContactsNumber(len(contactList));

        ''' Compute next momentum assuming no contacts '''
        self.M_inv      = np.linalg.inv(M);
        self.M_vp       = np.dot(M,v) - dt*h;
        self.M_vp[6:]   += dt*tau;
        self.v_p        = np.dot(self.M_inv, self.M_vp);
        
        if(self.k==0):
            return (self.v_p, np.zeros(0));
        
        self.f_old = np.copy(self.f0);
        while (self.rel_err>self.accuracy and self.iter<maxIter):
            self.N_alpha = self.compute_contact_projection(-self.M_vp-self.f_old);
            if(self.mu<1e-3):
                break;
            self.f_new   = self.compute_friction_projection(-self.M_vp-self.N_alpha);
            self.rel_err = self.compute_relative_error();
            self.residual = self.compute_residual();
            if(self.residual < self.min_res):
                self.min_res = self.residual;
                self.f_star = np.copy(self.f_new);  # cache best solution
            self.iter += 1;
            self.f_old = np.copy(self.f_new);
            
            if(self.verb>1):
                print "Iter %d err %f resid %f alpha %.3f beta %.3f" % (self.iter, self.rel_err, self.residual, 
                                                                    np.linalg.norm(self.alpha)/dt, 
                                                                    np.linalg.norm(self.beta)/dt);
        if(self.verb>0 and self.rel_err>self.accuracy):
            print "[Stag-Proj] Algorithm did not converge",
            print "Iter %d err %f resid %f alpha %.3f beta %.3f" % (self.iter, self.rel_err, self.residual, 
                                                                    np.linalg.norm(self.alpha)/dt, 
                                                                    np.linalg.norm(self.beta)/dt);
#            time.sleep(2);
                                                                    
        if(self.mu>1e-3):
            self.f0 = np.copy(self.f_star);
            self.N_alpha = self.compute_contact_projection(-self.M_vp-self.f0);
        v_next = self.v_p + np.dot(self.M_inv, self.N_alpha+self.f0);
        
        ddx = np.zeros(self.k);
        fz = np.zeros(self.k);
        v_xy = np.zeros(2*self.k);
        f_xy = np.zeros(2*self.k);
        for i in range(self.k):
            ''' compute normal contact accelerations and contact forces to check KKT conditions '''
            ddx[i] = np.dot(self.contactList[i][2,:], v_next) + dt*self.dJv_list[i][2];
            fz[i] = self.alpha[i]/dt;
        
            if(self.verb>0):
                ''' compute tangent velocities and forces '''
                v_xy[2*i:2*i+2] = np.dot(self.contactList[i][0:2,:], v_next);
                f_xy[2*i:2*i+2] = np.dot(self.T[0:2,:], self.beta[4*i:4*i+4])/dt;
                
                slide_acc = 1e-2;
                if(np.linalg.norm(v_xy[2*i:2*i+2])>slide_acc and fz[i]>10.0):
                    print "Contact point %d is sliding, fxy/fz=(%.2f,%.2f), fz=%.1f, v=(%.2f,%.2f,%.2f)"%(i,f_xy[2*i]/fz[i],f_xy[2*i+1]/fz[i],fz[i],
                                                                                                          v_xy[2*i],v_xy[2*i+1],ddx[i]);
#                    time.sleep(1);
             
            ''' compute contact forces '''
            self.contact_forces[3*i:3*i+3]  = np.dot(self.T, self.beta[4*i:4*i+4])/dt;
            self.contact_forces[3*i+2]      = self.alpha[i]/dt;

        self.v_xy = v_xy;
        self.f_xy = f_xy;
        
        if(np.dot(ddx,fz)>self.accuracy):
            print  'ERROR STAG-PROJ: ddx*fz = %f' % np.dot(ddx,fz);
            print ('                 ddx: ', ddx);
            print ('                 fz:  ', fz);

        self.computationTime        = time.time()-start;
        self.t += 1;
        self.meanComputationTime = ((self.t-1)*self.meanComputationTime + self.computationTime)/self.t;
        self.meanIterations      = ((self.t-1)*self.meanIterations + self.iter)/self.t;        
        if(self.maxComputationTime < self.computationTime):
            self.maxComputationTime = self.computationTime;
        if(self.maxIterations < self.iter):
            self.maxIterations = self.iter;
        
        return (v_next, self.contact_forces);
        
        
        
    def compute_contact_projection(self, z):        
        if(self.iter==0):
            for i in range(0,self.k):
                self.N[:,i] = self.contactList[i][2,:].T;        
                self.dJv[i] = self.dJv_list[i][2];
            self.H_contact      = np.dot(self.N.transpose(), np.dot(self.M_inv, self.N)) + 1e-10*np.identity(self.k);
            
        self.g_contact      = -np.dot(self.N.transpose(), np.dot(self.M_inv, z)) + self.dt*self.dJv;
        A = np.zeros((0,self.k));
        lbA = np.zeros(0);
        ubA = np.zeros(0);
        maxActiveSetIter    = np.array([2*self.k]);
        maxComputationTime  = np.array(self.maxTime);
        if(self.initialized_contact==False):
            imode = self.solverContact.init(self.H_contact, self.g_contact, A, self.lb_contact, self.ub_contact, 
                                            lbA, ubA, maxActiveSetIter, maxComputationTime);
            if(imode!=PyReturnValue.INIT_FAILED_INFEASIBILITY):
                self.initialized_contact = True;
        else:
            imode = self.solverContact.hotstart(self.H_contact, self.g_contact, A, self.lb_contact, self.ub_contact, 
                                            lbA, ubA, maxActiveSetIter, maxComputationTime);
        self.qpTime_contact     += maxComputationTime;
        self.iter_contact       += maxActiveSetIter[0];
        
        if(imode==0 or imode==PyReturnValue.MAX_NWSR_REACHED):
            self.solverContact.getPrimalSolution(self.alpha);

        self.print_qp_oases_error_message(imode, "Contact solver");
        
        return np.dot(self.N, self.alpha);
        
        
        
    def compute_friction_projection(self, z):                
        if(self.iter==0):
            self.T_dJ_dq = np.zeros(4*self.k);
            for i in range(0,self.k):
                self.D[:,4*i:4*i+4] = np.dot(self.contactList[i].T, self.T);
                self.T_dJ_dq[4*i:4*i+4] += self.dt * np.dot(self.T[0:2,:].T, self.dJv_list[i][0:2]);
            self.H_friction      = np.dot(self.D.transpose(), np.dot(self.M_inv, self.D));

        ''' compute constraint vector (upper bound) '''
        self.mu_alpha = self.mu * self.alpha;
        self.g_friction     = -np.dot(self.D.transpose(), np.dot(self.M_inv, z)) + self.T_dJ_dq;
        maxActiveSetIter    = np.array([8*self.k]);
        maxComputationTime  = np.array(self.maxTime);
        if(self.initialized_friction==False):
            imode = self.solverFriction.init(self.H_friction, self.g_friction, self.ET, self.lb_friction, self.ub_friction, 
                                             self.lbA_friction, self.mu_alpha, maxActiveSetIter, maxComputationTime);
            if(imode==0):
                self.initialized_friction = True;
        else:
            imode = self.solverFriction.hotstart(self.H_friction, self.g_friction, self.ET, self.lb_friction, self.ub_friction, 
                                             self.lbA_friction, self.mu_alpha, maxActiveSetIter, maxComputationTime);
        self.qpTime_friction     += maxComputationTime;
        self.iter_friction       += maxActiveSetIter[0];
        
        if(imode==0 or imode==PyReturnValue.MAX_NWSR_REACHED):
            self.solverFriction.getPrimalSolution(self.beta);

        self.print_qp_oases_error_message(imode, "Friction solver");
        
        return np.dot(self.D, self.beta);
                
                
                
    def compute_relative_error(self):
        f_diff = self.f_new - self.f_old;
        num = np.dot(f_diff.T, np.dot(self.M_inv, f_diff));
        den = np.dot(self.f_old.T, np.dot(self.M_inv, self.f_old));
        if(den!=0.0):
            return num/den;
        return 1e10;
            
            
            
    def compute_residual(self):
        res = 0.0;
        tmp = self.v_p + np.dot(self.M_inv, self.N_alpha + self.f_new);
        for i in range(self.k):
            res += abs(self.alpha[i]*np.dot(self.N[:,i].T, tmp));
        return res;
            
            
            
    def reset(self):
        self.initialized_contact = False;
        self.initialized_friction = False;



    def print_qp_oases_error_message(self, imode, solver_name):
        if(imode!=0 and imode!=63 and self.verb>0):
            if(imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY):
                print "[%s] ERROR Qp oases HOTSTART_STOPPED_INFEASIBILITY" % solver_name; # 60
            elif(imode==PyReturnValue.MAX_NWSR_REACHED):
                print "[%s] ERROR Qp oases RET_MAX_NWSR_REACHED" % solver_name; # 63
            elif(imode==PyReturnValue.STEPDIRECTION_FAILED_CHOLESKY):
                print "[%s] ERROR Qp oases STEPDIRECTION_FAILED_CHOLESKY" % solver_name; # 68
            elif(imode==PyReturnValue.HOTSTART_FAILED_AS_QP_NOT_INITIALISED):
                print "[%s] ERROR Qp oases HOTSTART_FAILED_AS_QP_NOT_INITIALISED" % solver_name; # 53
            elif(imode==PyReturnValue.INIT_FAILED_INFEASIBILITY):
                print "[%s] ERROR Qp oases INIT_FAILED_INFEASIBILITY" % solver_name; # 37
#                    RET_INIT_FAILED_HOTSTART = 36
            elif(imode==PyReturnValue.UNKNOWN_BUG):
                print "[%s] ERROR Qp oases UNKNOWN_BUG" % solver_name; # 9
            else:
                print "[%s] ERROR Qp oases %d " % (solver_name, imode);