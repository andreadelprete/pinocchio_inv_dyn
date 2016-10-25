import numpy as np
from qpoases import PyQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
import time
from math import sqrt
from sot_utils import crossMatrix, qpOasesSolverMsg
from numpy.linalg import norm

EPS = 1e-5

class ComAccPP(object):
    
    def __init__(self, name, c0, dc0, contact_points, contact_normals, mu, g, mass, verb=0):
        self.name = name;
        (H,h) = compute_GIWC(contact_points, contact_normals, mu, False, True);
        ''' Project GIWC in (alpha,DDalpha) space, where c=c0+alpha*v, ddc=DDalpha*v, v=dc0/||dc0||: a*alpha + b*DDalpha <= d '''
        v = dc0/norm(dc0);
        K = np.zeros((6,3));
        K[:3,:] = mass*np.identity(3);
        K[3:,:] = mass*crossMatrix(c0);
        self.d = h - np.dot(H, np.dot(K, g));
        self.b = np.dot(np.dot(-H,K),v);             # vector multiplying com acceleration
        tmp = np.array([[0,0,0,0,1,0],          #temp times the variation dc will result in 
                        [0,0,0,-1,0,0],         # [ 0 0 0 dc_y -dc_x 0]^T
                        [0,0,0,0,0,0]]).T;   
        self.a = mass*norm(g)*np.dot(np.dot(H,tmp),v);
        
        ''' Normalize inequalities to have unitary coefficients for DDalpha: b*DDalpha <= d - a*alpha '''
        for i in range(self.a.shape[0]):
            if(abs(self.b[i]) > EPS):
                self.a[i] /= abs(self.b[i]);
                self.d[i] /= abs(self.b[i]);
                self.b[i] /= abs(self.b[i]);
        
    def compute_max_deceleration(self, alpha):
        imode = 0;
        ''' Find current active inequality: b*DDalpha <= d - a*alpha (i.e. DDalpha lower bound for current alpha value) ''' 
        #sort b indices to only keep negative values
        negative_ids = np.where(self.b<0)[0];
        a_alpha_d = self.a*alpha-self.d;
        a_alpha_d_negative_bs = a_alpha_d[negative_ids];
        (i_DDalpha_min, DDalpha_min) = [(i,a_min) for (i, a_min) in [(j, a_alpha_d[j]) for j in negative_ids] if (a_min >= a_alpha_d_negative_bs).all()][0];
        ''' Find alpha_max (i.e. value of alpha corresponding to right vertex of active inequality) '''
        den = self.b*self.a[i_DDalpha_min] + self.a;
        i_pos = np.where(den>0)[0];
        if(i_pos.shape[0]==0):
            print "[%s] ERROR b*a_i0+a is never positive, that means that alpha_max is unbounded"%self.name, den;
            alpha_max = alpha;
            imode = 1;
        else:
            alpha_max = np.min((self.d[i_pos] + self.b[i_pos]*self.d[i_DDalpha_min])/den[i_pos]);
            if(alpha_max<alpha):
                print "[%s] ERROR: alpha_max<alpha"%self.name, alpha_max, alpha;
                imode = 2;
        return (imode, DDalpha_min, self.a[i_DDalpha_min], alpha_max);
            
    def plotFeasibleAccPolytope(self, range_plot=2.0):
        ax = plot_inequalities(np.vstack([self.a,self.b]).T, self.d, [-range_plot,range_plot], [-range_plot,range_plot]);
        plt.axis([0,range_plot,-range_plot,0])
        plt.title('Feasible com pos-acc');
        plut.movePlotSpines(ax, [0, 0]);
        ax.set_xlabel('com pos');
        ax.set_ylabel('com acc');
        plt.show();
            


class ComAccLP (object):
    """
    LP solver dedicated to finding the maximum center of mass (CoM) deceleration in a 
    specified direction, subject to the friction cone constraints.
    This is possible thanks to the simplifying assumption that the CoM acceleration
    is going to be parallel to its velocity. This allows us to represent the 3d
    com trajectory by means of a 1d trajectory alpha(t):
        c(t) = c0 + alpha(t) v
        v = c0 / ||c0||
        dc = dAlpha(t) v
        ddc(t) = ddAlpha(t) v
    The operation amounts to solving the following parametric Linear Program:
      minimize      ddAlpha
      subject to    A f = a ddAlpha + b alpha + d
                    B f <= 0
    where:
      f         are the contact forces
      ddAlpha   is the magnitude of the CoM acceleration
      alpha     is the magnitude of the CoM displacement (with respect to its initial position c0)
    Given a CoM position (by means of a value of alpha), this class can compute the 
    minimum com acceleration in direction v (i.e. the maximum acceleration in direction -v).
    Since this is a piecewise-linear function of alpha, it can also compute its derivative with respect 
    to alpha, and the boundaries of the alpha-region in which the derivative remains constant.
    """
    
    NO_WARM_START = False;
    
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of inequalities
    
    Hess = [];     # Hessian  H = G^T*G
    grad = [];     # gradient
    
    solver='';  # type of solver to use
    accuracy=0; # accuracy used by the solver for termination
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

    def __init__ (self, name, c0, dc0, contact_points, contact_normals, mu, g, mass, maxIter=10000, verb=0, regularization=1e-10):
        assert contact_points.shape[1]==3
        assert contact_normals.shape[1]==3
        assert contact_points.shape[0]==contact_normals.shape[0]
        contact_points = np.asarray(contact_points);
        contact_normals = np.asarray(contact_normals);
        nContacts = contact_points.shape[0];
        if(isinstance(mu, (list, tuple, np.ndarray))):
            mu = np.asarray(mu).squeeze();
        else:
            mu = mu*np.ones(nContacts);
        ''' compute generators '''
        cg = 4;
        nGen = nContacts*cg;           # number of generators
        G4 = np.zeros((3,cg));
        self.A = np.zeros((6,nGen));
        P = np.zeros((6,3));
        P[:3,:] = -np.identity(3);
        muu = mu/sqrt(2.0);
        for i in range(nContacts):
            ''' compute tangent directions '''
            contact_normals[i,:]  = contact_normals[i,:]/norm(contact_normals[i,:]);
            T1 = np.cross(contact_normals[i,:], [0.,1.,0.]);
            if(norm(T1)<EPS):
                T1 = np.cross(contact_normals[i,:], [1.,0.,0.]);
            T1 = T1/norm(T1);
            T2 = np.cross(contact_normals[i,:], T1);
            G4[:,0] =  muu[i]*T1 + muu[i]*T2 + contact_normals[i,:];
            G4[:,1] =  muu[i]*T1 - muu[i]*T2 + contact_normals[i,:];
            G4[:,2] = -muu[i]*T1 + muu[i]*T2 + contact_normals[i,:];
            G4[:,3] = -muu[i]*T1 - muu[i]*T2 + contact_normals[i,:];
            ''' normalize generators '''
            for j in range(cg):
                G4[:,j] /= norm(G4[:,j]);
            ''' compute matrix mapping contact forces to gravito-inertial wrench '''
            P[3:,:] = -crossMatrix(contact_points[i,:]);
            ''' project generators in 6d centroidal space '''
            self.A[:,cg*i:cg*i+cg] = np.dot(P, G4);

        self.v = dc0 / norm(dc0);
        self.a = np.empty(6);
        self.a[:3] = -mass*self.v;
        self.a[3:] = -mass*np.cross(c0, self.v);
        self.b = np.zeros(6);
        self.b[3:] = mass*np.cross(self.v, g);
        self.d = np.empty(6);
        self.d[:3] = mass*g;
        self.d[3:] = mass*np.cross(c0, g);

        self.name = name;        
        self.iter       = 0;
        self.maxIter    = maxIter;
        self.verb       = verb;
        self.m_in       = 6;
        self.n          = nGen+1;
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
        self.options.enableRegularisation = False;
        self.qpOasesSolver.setOptions(self.options);
        self.initialized = False;
#        self.qpOasesSolver.printOptions()

        self.Hess = 1e-10*np.identity(self.n);
        self.grad = np.ones(self.n)*regularization;
        self.grad[-1] = 1.0;
        self.constrMat = np.zeros((self.m_in,self.n));
        self.constrMat[:,:-1] = self.A;
        self.constrMat[:,-1] = -self.a;
        self.constrUB = np.zeros(self.m_in)+1e100;
        self.constrLB = np.zeros(self.m_in)-1e100;
        self.lb = np.zeros(self.n);
        self.lb[-1] = -1e100;
        self.ub = np.array(self.n*[1e100,]);
        self.x  = np.zeros(self.n);
        self.y  = np.zeros(self.n+self.m_in);
        

    def compute_max_deceleration(self, alpha, maxIter=None, maxTime=100.0):
        start = time.time();
        if(self.NO_WARM_START):
            self.qpOasesSolver  = SQProblem(self.n,self.m_in);
            self.qpOasesSolver.setOptions(self.options);
            self.initialized = False;            
        if(maxIter==None):
            maxIter = self.maxIter;        
        maxActiveSetIter    = np.array([maxIter]);
        maxComputationTime  = np.array(maxTime);
        self.constrUB[:6]   = np.dot(self.b, alpha) + self.d;
        self.constrLB[:6]   = self.constrUB[:6];
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
        if(self.imode==0):
            self.qpOasesSolver.getPrimalSolution(self.x);
            self.qpOasesSolver.getDualSolution(self.y);

            if((self.x<self.lb-EPS).any()):
                print "[%s] ERROR lower bound violated" % (self.name), self.x, self.lb;
            if((self.x>self.ub+EPS).any()):
                print "[%s] ERROR upper bound violated" % (self.name), self.x, self.ub;
            if((np.dot(self.constrMat,self.x)>self.constrUB+EPS).any()):
                print "[%s] ERROR constraint upper bound violated" % (self.name), np.dot(self.constrMat,self.x), self.constrUB;
            if((np.dot(self.constrMat,self.x)<self.constrLB-EPS).any()):
                print "[%s] ERROR constraint lower bound violated" % (self.name), np.dot(self.constrMat,self.x), self.constrLB;
        
            act_set = np.where(self.y[:self.n-1]!=0.0)[0];    # indexes of active bound constraints
            n_as = act_set.shape[0];
            if(n_as > self.n-6):
                raise ValueError("[%s] ERROR Too many active constraints: %d (rather than %d)" % (self.name, n_as, self.n-6));
            elif(self.verb>0 and n_as < self.n-6):
                print "[%s] WARNING Less active constraints than expected: %d (rather than %d)" % (self.name, n_as, self.n-6);
            self.K = np.zeros((n_as+6,self.n));
            self.k1 = np.zeros(n_as+6);
            self.k2 = np.zeros(n_as+6);
            self.K[:n_as,:] = np.identity(self.n)[act_set,:];
            self.K[-6:,:-1] = self.A;
            self.K[-6:,-1] = -self.a;            
            self.k1[-6:] = self.b;            
            self.k2[-6:] = self.d;
            U, s, VT = np.linalg.svd(self.K);
            rank = (s > EPS).sum();
            K_inv = np.dot(VT[:rank,:].T, np.dot(np.diag(1/s[:rank]), U[:,:rank].T));
            K_inv_k1 = np.dot(K_inv, self.k1);
            K_inv_k2 = np.dot(K_inv, self.k2);
            if(rank<self.n):
#                print "WARNING Matrix K is singular: rank=", rank;
                Z = VT[rank:,:].T;
                P = np.dot(np.dot(Z, np.linalg.inv(np.dot(Z.T, np.dot(self.Hess, Z)))), Z.T);
                K_inv_k1 -= np.dot(P, np.dot(self.Hess, K_inv_k1));
                K_inv_k2 -= np.dot(P, np.dot(self.Hess, K_inv_k2) + self.grad);
                
            # Check that the solution you get by solving the KKT is the same found by the solver
            x_kkt = K_inv_k1*alpha + K_inv_k2;
            if(norm(self.x - x_kkt) > EPS):
                raise ValueError("[%s] ERROR x different from x_kkt. x=" % (self.name)+str(self.x)+"\nx_kkt="+str(x_kkt)+" "+str(norm(self.x - x_kkt)));
            dx = K_inv_k1[-1];
            # act_set_mat * alpha >= act_set_vec
            act_set_mat = K_inv_k1[:-1];
            act_set_vec = -K_inv_k2[:-1];
            if (act_set_mat*alpha<act_set_vec-EPS).any():
                raise ValueError("ERROR: current alpha violates constraints "+str(act_set_mat*alpha-act_set_vec));
            for i in range(act_set_mat.shape[0]):
                if(abs(act_set_mat[i])>EPS):
                    act_set_vec[i] /= abs(act_set_mat[i]);
                    act_set_mat[i] /= abs(act_set_mat[i]);
            if (act_set_mat*alpha<act_set_vec-EPS).any():
                raise ValueError("ERROR: after normalization current alpha violates constraints "+str(act_set_mat*alpha-act_set_vec));
            alpha_min = np.max(act_set_vec[act_set_mat>EPS]);
            alpha_max = np.min(-act_set_vec[act_set_mat<-EPS]);
            if(alpha_min > alpha_max):
                raise ValueError("ERROR alpha_min %.3f > alpha_max %.3f" % (alpha_min,alpha_max));
            
        elif(self.verb>0):
            print "[%s] ERROR Qp oases %s" % (self.name, qpOasesSolverMsg(self.imode));
            dx=0.0;
            alpha_min=0.0;
            alpha_max=0.0;
        if(self.qpTime>=maxTime):
            if(self.verb>0):
                print "[%s] Max time reached %f after %d iters" % (self.name, self.qpTime, self.iter);
            self.imode = 9;   
        self.computationTime        = time.time()-start;        
        return (self.imode, self.x[-1], dx, alpha_min, alpha_max);

                    
    def reset(self):
        self.initialized = False;
    
        

from multi_contact_stability_criterion_utils import generate_contacts, compute_GIWC, find_static_equilibrium_com
import plot_utils as plut
from geom_utils import plot_inequalities
import matplotlib.pyplot as plt
from math import atan, pi
import cProfile
        
def test():
    np.set_printoptions(precision=2, suppress=True);
    DO_PLOTS = False;
    PLOT_3D = False;
    mass = 75;             # mass of the robot
    g_vector = np.array([0,0,-9.81]);
    mu = 0.5;           # friction coefficient
    lx = 0.1;           # half foot size in x direction
    ly = 0.07;          # half foot size in y direction
    USE_DIAGONAL_GENERATORS = True;
    GENERATE_QUASI_FLAT_CONTACTS = True;
    #First, generate a contact configuration
    CONTACT_POINT_UPPER_BOUNDS = [ 0.5,  0.5,  0.5];
    CONTACT_POINT_LOWER_BOUNDS = [-0.5, -0.5,  0.0];
    gamma = atan(mu);   # half friction cone angle
    RPY_LOWER_BOUNDS = [-2*gamma, -2*gamma, -pi];
    RPY_UPPER_BOUNDS = [+2*gamma, +2*gamma, +pi];
    MIN_CONTACT_DISTANCE = 0.3;
    N_CONTACTS = 2
    
    X_MARG = 0.07;
    Y_MARG = 0.07;
    
    succeeded = False;
    
    while(succeeded == False):
        (p, N) = generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, GENERATE_QUASI_FLAT_CONTACTS);        
        X_LB = np.min(p[:,0]-X_MARG);
        X_UB = np.max(p[:,0]+X_MARG);
        Y_LB = np.min(p[:,1]-Y_MARG);
        Y_UB = np.max(p[:,1]+Y_MARG);
        Z_LB = np.min(p[:,2]-0.05);
        Z_UB = np.max(p[:,2]+1.5);
        (H,h) = compute_GIWC(p, N, mu, False, USE_DIAGONAL_GENERATORS);
        (succeeded, c0) = find_static_equilibrium_com(mass, [X_LB, Y_LB, Z_LB], [X_UB, Y_UB, Z_UB], H, h);
        
    dc0 = np.random.uniform(-1, 1, size=3); 
    dc0[2] = 0;
    
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
        plt.axis([X_LB,X_UB,Y_LB,Y_UB]);           
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
   
    comAccLP = ComAccLP("comAccLP", c0, dc0, p, N, mu, g_vector, mass, maxIter=100, verb=1, regularization=1e-5);
    comAccLP2 = ComAccLP("comAccLP", c0, dc0, p, N, mu, g_vector, mass, maxIter=100, verb=1, regularization=1e-10);
    comAccPP = ComAccPP("comAccPP", c0, dc0, p, N, mu, g_vector, mass, verb=1);
    alpha = 0.0;
    ddAlpha = -1.0;
    while(ddAlpha<0.0):
        (imode, ddAlpha, slope, alpha_min, alpha_max) = comAccLP.compute_max_deceleration(alpha);
        if(imode==0):
            print "ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f, qp time %.3f)"%(alpha,ddAlpha,slope,alpha_max,1e3*comAccLP.qpTime);
        else:
            print "LP failed!"

        (imode2, ddAlpha2, slope2, alpha_max2) = comAccPP.compute_max_deceleration(alpha);            
        if(imode2==0):
            print "ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f)"%(alpha,ddAlpha2,slope2,alpha_max2);
        else:
            print "PP failed!";
            
        if(imode!= 0 or imode2!=0):
            break;
        if(abs(ddAlpha-ddAlpha2) > 1e-4):
            print "\n *** ERROR ddAlpha", ddAlpha-ddAlpha2;
        if(abs(slope-slope2) > 1e-3):
            print "\n *** ERROR slope", slope-slope2;
        if(abs(alpha_max-alpha_max2) > 1e-3):
            if(alpha_max < alpha_max2):
                print "\n *** WARNING alpha_max", alpha_max-alpha_max2;
            else:
                print "\n *** ERROR alpha_max", alpha_max-alpha_max2;
        alpha = alpha_max+EPS;
        
    print "\n\n\n SECOND TEST \n\n\n"
    alpha = 0.0;
    ddAlpha = -1.0;
    while(ddAlpha<0.0):
        (imode, ddAlpha, slope, alpha_min, alpha_max) = comAccLP2.compute_max_deceleration(alpha);
        if(imode==0):
            print "ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f, qp time %.3f)"%(alpha,ddAlpha,slope,alpha_max,1e3*comAccLP2.qpTime);
        else:
            print "LP failed!"

        (imode2, ddAlpha2, slope2, alpha_max2) = comAccPP.compute_max_deceleration(alpha);            
        if(imode2==0):
            print "ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f)"%(alpha,ddAlpha2,slope2,alpha_max2);
        else:
            print "PP failed!";
            
        if(imode!= 0 or imode2!=0):
            break;
        if(abs(ddAlpha-ddAlpha2) > 1e-4):
            print "\n *** ERROR ddAlpha", ddAlpha-ddAlpha2;
        if(abs(slope-slope2) > 1e-3):
            print "\n *** ERROR slope", slope-slope2;
        if(abs(alpha_max-alpha_max2) > 1e-3):
            if(alpha_max < alpha_max2):
                print "\n *** WARNING alpha_max", alpha_max-alpha_max2;
            else:
                print "\n *** ERROR alpha_max", alpha_max-alpha_max2;
        alpha = alpha_max+EPS;

    if(DO_PLOTS):
        comAccPP.plotFeasibleAccPolytope();
    


if __name__=="__main__":
    for i in range(1):
        try:
            test();
#            ret = cProfile.run("test()");
        except Exception as e:
            print e;
            continue;
