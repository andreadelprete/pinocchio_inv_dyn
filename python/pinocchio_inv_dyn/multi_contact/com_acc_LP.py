import numpy as np
from numpy.linalg import norm
import pinocchio_inv_dyn.optimization.solver_LP_abstract as optim #import optim.getNewSolver, LP_status, LP_status_string
from pinocchio_inv_dyn.sot_utils import crossMatrix
from pinocchio_inv_dyn.multi_contact.utils import generate_contacts, compute_GIWC, find_static_equilibrium_com, compute_centroidal_cone_generators
import pinocchio_inv_dyn.plot_utils as plut
import matplotlib.pyplot as plt
import warnings


EPS = 1e-5;
INITIAL_HESSIAN_REGULARIZATION = 1e-8;
MAX_HESSIAN_REGULARIZATION = 1e-4;
FORCE_GENERATOR_COEFFICIENT_UPPER_BOUND = 1e5;

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
      minimize      ddAlpha + w sum_i(f_i)
      subject to    A f = a ddAlpha + b alpha + d
                    f >= 0
    where:
      f         are the contact forces generator coefficients
      ddAlpha   is the magnitude of the CoM acceleration
      alpha     is the magnitude of the CoM displacement (with respect to its initial position c0)
      w         regularization parameter
    Given a CoM position (by means of a value of alpha), this class can compute the 
    minimum com acceleration in direction v (i.e. the maximum acceleration in direction -v).
    Since this is a piecewise-linear function of alpha, it can also compute its derivative with respect 
    to alpha, and the boundaries of the alpha-region in which the derivative remains constant.
    """
    
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of constraints (i.e. 6)
    
    grad = [];     # gradient
    A = None;       # constraint matrix multiplying the contact force generators
    b = None;       # constraint vector multiplying the CoM position parameter alpha
    d = None;       # constraint vector
    
    mass = 0.0; # robot mass
    g = None;   # 3d gravity vector
    verb=0;     # verbosity level of the solver (0=min, 2=max)
    
    solver = None;
    
    def __init__ (self, name, c0, v, contact_points, contact_normals, mu, g, mass, 
                  contact_tangents=None, maxIter=10000, verb=0, regularization=1e-5, solver='qpoases'):
        ''' Constructor
            @param c0 Initial CoM position
            @param v Opposite of the direction in which you want to maximize the CoM acceleration (typically that would be
                                                                                                   the CoM velocity direction)
            @param g Gravity vector
            @param contact_tangents Nx3 matrix containing the contact tangents
            @param regularization Weight of the force minimization, the higher this value, the sparser the solution
            @param solver Specify which LP solver to use (qpoases, cvxopt or scipy)
        '''
        self.name       = name;
        self.verb       = verb;
        self.m_in       = 6;
        self.solver = optim.getNewSolver(solver, name, maxIter=maxIter, verb=verb);
        self.solver.setUseWarmStart(False);
        self.hessian_regularization = INITIAL_HESSIAN_REGULARIZATION;
        self.b = np.zeros(6);
        self.d = np.empty(6);
        self.c0 = np.empty(3);
        self.v = np.empty(3);
        self.constrUB = np.zeros(self.m_in)+1e100;
#        self.constrLB = np.zeros(self.m_in)-1e100;
        self.set_problem_data(c0, v, contact_points, contact_normals, mu, g, mass, contact_tangents, regularization);


    def set_com_state(self, c0, v):
        assert np.asarray(c0).squeeze().shape[0]==3, "Com position vector has not size 3"
        assert np.asarray(v).squeeze().shape[0]==3, "Com acceleration direction vector has not size 3"
        self.c0 = np.asarray(c0).squeeze();
        self.v = np.asarray(v).squeeze().copy();
        if(norm(self.v)==0.0):
            raise ValueError("[%s] Norm of com acceleration direction v is zero!"%self.name);
        self.v /= norm(self.v);

        self.constrMat[:3,-1] = self.mass*self.v;
        self.constrMat[3:,-1] = self.mass*np.cross(self.c0, self.v);
        self.b[3:] = self.mass*np.cross(self.v, self.g);
        self.d[:3] = self.mass*self.g;
        self.d[3:] = self.mass*np.cross(c0, self.g);


    def set_contacts(self, contact_points, contact_normals, mu, contact_tangents=None, regularization=1e-5):
        # compute matrix A, which maps the force generator coefficients into the centroidal wrench
        (self.A, self.G4) = compute_centroidal_cone_generators(contact_points, contact_normals, mu, contact_tangents);
        
        # since the size of the problem may have changed we need to recreate the solver and all the problem matrices/vectors
        if(self.n != contact_points.shape[0]*4 + 1):
            self.n              = contact_points.shape[0]*4 + 1;
            self.constrMat = np.zeros((self.m_in,self.n));
            self.constrMat[:3,-1] = self.mass*self.v;
            self.constrMat[3:,-1] = self.mass*np.cross(self.c0, self.v);
            self.lb = np.zeros(self.n);
            self.lb[-1] = -1e3;
            self.ub = np.ones(self.n)*FORCE_GENERATOR_COEFFICIENT_UPPER_BOUND;
            self.x  = np.zeros(self.n);
            self.y  = np.zeros(self.n+self.m_in);
            self.grad = np.ones(self.n);
            self.grad[-1] = 1.0;

        self.grad[:-1] = regularization;
        self.constrMat[:,:-1] = self.A;


    def set_problem_data(self, c0, v, contact_points, contact_normals, mu, g, mass, contact_tangents=None, regularization=1e-5):
        assert np.asarray(g).squeeze().shape[0]==3, "Gravity vector has not size 3"
        assert mass>0.0, "Mass is not positive"
        self.mass = mass;
        self.g = np.asarray(g).squeeze().copy();
        self.set_contacts(contact_points, contact_normals, mu, contact_tangents, regularization);
        self.set_com_state(c0, v);

    def compute_x_kkt(self):
        act_set = np.where(self.y[:self.n-1]!=0.0)[0];    # indexes of active bound constraints
        n_as = act_set.shape[0];
        if(n_as > self.n-6):
            raise ValueError("[%s] ERROR Too many active constraints: %d (rather than %d)" % (self.name, n_as, self.n-6));
        if(self.verb>1 and n_as < self.n-6):
            print "[%s] INFO Less active constraints than expected: %d (rather than %d)" % (self.name, n_as, self.n-6);
        self.K = np.zeros((n_as+6,self.n));
        self.k1 = np.zeros(n_as+6);
        self.k2 = np.zeros(n_as+6);
        self.K[:n_as,:] = np.identity(self.n)[act_set,:];
        self.K[-6:,:] = self.constrMat;
        act_ub = np.where(self.x[act_set]==self.ub[act_set])[0];
        if(self.verb>-1 and act_ub.shape[0]>0):
            print "[%s] INFO %d contact forces saturated upper bound"%(self.name, act_ub.shape[0]);
        self.k2[act_ub] = FORCE_GENERATOR_COEFFICIENT_UPPER_BOUND; #self.ub[act_set[act_ub]];
        self.k1[-6:] = self.b;            
        self.k2[-6:] = self.d;
        U, s, VT = np.linalg.svd(self.K);
        rank = int((s > EPS).sum());
        self.K_inv_k1 = np.dot(VT[:rank,:].T, (1.0/s[:rank])*np.dot(U[:,:rank].T, self.k1));
        self.K_inv_k2 = np.dot(VT[:rank,:].T, (1.0/s[:rank])*np.dot(U[:,:rank].T, self.k2));
        
        if(rank<self.n):
            # solution is not at a vertex of the feasible polytope (this would never happen with an LP)
            Z = VT[rank:,:].T;
            P = np.dot(np.dot(Z, np.linalg.inv(self.hessian_regularization*np.dot(Z.T, Z))), Z.T);
            self.K_inv_k1 -= np.dot(P, self.hessian_regularization*self.K_inv_k1);
            self.K_inv_k2 -= np.dot(P, self.hessian_regularization*self.K_inv_k2 + self.grad);
        
        if(rank<self.K.shape[0]):
            # active constraints are not linearly dependent, so infinite solutions to the KKT conditions exist
            if(self.verb>-1):
                print "*** [%s] INFO Active constraints are not linearly dependent (rank %d out of %d)"%(self.name,rank,self.K.shape[0])
            x_kkt = self.K_inv_k1*self.alpha + self.K_inv_k2;
            # make sure that you get the same solution found by the solver by solving the KKT conditions
            self.K_inv_k2 += self.x-x_kkt;
            
        # compute the solution by solving the KKT conditions
        x_kkt = self.K_inv_k1*self.alpha + self.K_inv_k2;
        return x_kkt;

    def compute_max_deceleration_derivative(self):
        ''' Compute the derivative of the max CoM deceleration (i.e. the solution of the last LP)
            with respect to the parameter alpha (i.e. the CoM position parameter). Moreover, 
            it also computes the bounds within which this derivative is valid (alpha_min, alpha_max).
        '''

        # store the derivative of the solution w.r.t. the parameter alpha
        dx = self.K_inv_k1[-1];
        # act_set_mat * alpha >= act_set_vec
        act_set_mat = self.K_inv_k1[:-1];
        act_set_vec = -self.K_inv_k2[:-1];
        if (act_set_mat*self.alpha<act_set_vec-EPS).any():
            raise ValueError("ERROR: current alpha violates constraints "+str(np.min(act_set_mat*self.alpha-act_set_vec))+
                             " min(x_kkt)="+str(np.min(self.K_inv_k1[:-1]*self.alpha+self.K_inv_k2[:-1])));
        for i in range(act_set_mat.shape[0]):
            if(abs(act_set_mat[i])>EPS):
                act_set_vec[i] /= abs(act_set_mat[i]);
                act_set_mat[i] /= abs(act_set_mat[i]);
        if (act_set_mat*self.alpha<act_set_vec-EPS).any():
            raise ValueError("ERROR: after normalization current alpha violates constraints "+str(np.min(act_set_mat*self.alpha-act_set_vec)));

        ind_pos = np.where(act_set_mat>EPS)[0];
        if(ind_pos.shape[0]>0):
            alpha_min = np.max(act_set_vec[ind_pos]);
        else:
            alpha_min = -1e10;
#            warnings.warn("[%s] alpha_min seems unbounded %.7f"%(self.name, np.max(act_set_mat)));

        ind_neg = np.where(act_set_mat<-EPS)[0];
        if(ind_neg.shape[0]>0):
            alpha_max = np.min(-act_set_vec[ind_neg]);
        else:
            alpha_max = 1e10;
#            warnings.warn("[%s] alpha_max seems unbounded %.7f"%(self.name, np.min(act_set_mat)));

        if(alpha_min > alpha_max):
            raise ValueError("ERROR alpha_min %.3f > alpha_max %.3f" % (alpha_min,alpha_max));
        return (dx, alpha_min, alpha_max);


    def compute_max_deceleration(self, alpha):
        self.alpha = alpha;
        self.constrUB[:6]   = np.dot(self.b, alpha) + self.d;
#        self.constrLB[:6]   = self.constrUB[:6];

        while(True):
            (status, self.x, self.y) = self.solver.solve(self.grad, self.lb, self.ub, 
                                                         A_eq=self.constrMat, b=self.constrUB);
            
            if(status==optim.LP_status.OPTIMAL):
                x_kkt = self.compute_x_kkt();
                if(norm(self.x - x_kkt) < 10*EPS):
                    break;
                if(self.verb>1):
                    print "[%s] INFO Solver succeeded but x!=x_kkt (%f)" % (self.name, norm(self.x-x_kkt));
                if(self.hessian_regularization>=MAX_HESSIAN_REGULARIZATION):
                    warnings.warn("[%s] ERROR x different from x_kkt. |x-x_kkt|=%f" % (self.name, norm(self.x - x_kkt))+ 
                                  ", x="+str(self.x)+", y="+str(self.y)+", active constraints="+str(np.where(self.y[:self.n-1]!=0.0)[0].shape[0])+
                                  ", expected active constraints="+str(self.n-6));
                    if(self.verb>0):
                        self._debug_kkt();
                    
            if(status==optim.LP_status.INFEASIBLE or self.hessian_regularization>=MAX_HESSIAN_REGULARIZATION):
                break;
                
            self.solver.reset();
            self.hessian_regularization *= 10.0;
            if(not self.solver.set_option('hessian_regularization', self.hessian_regularization)):
                break;
            if(self.verb>0):
                print "[%s] WARNING %s. Increasing Hessian regularization to %f"%(self.name, optim.LP_status_string[status], self.hessian_regularization);
            
        if(status==optim.LP_status.OPTIMAL):
            (dx, alpha_min, alpha_max) = self.compute_max_deceleration_derivative();
        else:
            self.initialized = False;
            dx=0.0;
            alpha_min=0.0;
            alpha_max=0.0;
        return (status, self.x[-1], dx, alpha_min, alpha_max);


    def getContactForces(self):
        ''' Get the contact forces obtained by solving the last LP '''
        cg = 4;
        nContacts = self.G4.shape[1]/cg;
        f = np.empty((3,nContacts));
        for i in range(nContacts):
            f[:,i] = np.dot(self.G4[:,cg*i:cg*i+cg], self.x[cg*i:cg*i+cg]);
        return f;

                    
    def reset(self):
        self.initialized = False;
        
    def getLpTime(self):
        return self.solver.getLpTime();
        
    def _debug_kkt(self):
        maxStat = np.zeros(1); # maximum value of stationarity condition residual.
        maxFeas = np.zeros(1); # maximum value of primal feasibility violation.
        maxCmpl = np.zeros(1); # maximum value of complementarity residual. 
        self.analyser.getKktViolation(self.qpOasesSolver, maxStat, maxFeas, maxCmpl);
        print("maxStat: %e, maxFeas:%e, maxCmpl: %e\n"%(maxStat, maxFeas, maxCmpl));
        
        act_set = np.where(self.y!=0.0)[0];    # indexes of active bound constraints
        kkt_viol = np.dot(self.Hess, self.x) + self.grad - np.dot(self.K.T, self.y[act_set]);
        print "kkt_viol   =", norm(kkt_viol);
        constr_viol = np.dot(self.K, self.x) - self.k1*self.alpha - self.k2;
        print "constr_viol=", norm(constr_viol);

        n_as = act_set.shape[0];
        KKT = np.zeros((self.n+n_as, self.n+n_as));
        KKT[:self.n, :self.n] = self.Hess;
        KKT[:self.n, self.n:] = -self.K.T;
        KKT[self.n:, :self.n] = self.K;
        
        k1_bar = np.zeros(self.n+n_as);
        k1_bar[-n_as:] = self.k1;
        k2_bar = np.zeros(self.n+n_as);
        k2_bar[:self.n] = -self.grad;
        k2_bar[self.n:] = self.k2;
        
        xy = np.zeros(self.n+n_as);
        xy[:self.n] = self.x;
        xy[self.n:] = self.y[act_set];
        k = k1_bar*self.alpha + k2_bar;
        print "KKT*xy-k=", norm(np.dot(KKT, xy) - k);

        U, s, VT = np.linalg.svd(KKT);
        rank = int((s > EPS).sum());                    
        xy_kkt = np.dot(VT[:rank,:].T, (1.0/s[:rank])*np.dot(U[:,:rank].T, k));
        print "rank(KKT)", rank, "dim", self.n+n_as
        print "smallest sing val", s[-1]
        print "|x-x_kkt|="+str(norm(self.x-xy_kkt[:self.n]));

        U, s, VT = np.linalg.svd(self.K);
        rank = int((s > EPS).sum());  
        print "rank(K)=%d, dim=%d"%(rank, n_as);
    

class ComAccPP(object):
    
    def __init__(self, name, c0, dc0, contact_points, contact_normals, mu, g, mass, verb=0):
        self.name = name;
        self.verb = verb;
        (H,h) = compute_GIWC(contact_points, contact_normals, mu);
        ''' Project GIWC in (alpha,DDalpha) space, where c=c0+alpha*v, ddc=DDalpha*v, v=dc0/||dc0||: a*alpha + b*DDalpha <= d '''
        v = dc0/norm(dc0);
        K = np.zeros((6,3));
        K[:3,:] = mass*np.identity(3);
        K[3:,:] = mass*crossMatrix(c0);
        self.d = h - np.dot(H, np.dot(K, g));
        self.b = np.dot(np.dot(-H,K),v);        # vector multiplying com acceleration
        tmp = np.array([[0,0,0,0,1,0],          #tmp times the variation dc will result in 
                        [0,0,0,-1,0,0],         # [ 0 0 0 dc_y -dc_x 0]^T
                        [0,0,0,0,0,0]]).T;   
        self.a = mass*norm(g)*np.dot(np.dot(H,tmp),v);
        
#        ''' Eliminate redundant inequalities '''
#        if(eliminate_redundancies):
#            A_red, self.d = eliminate_redundant_inequalities(np.vstack([self.a,self.b]).T, self.d);
#            self.a = A_red[:,0];
#            self.b = A_red[:,1];

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
        negative_ids = np.where(self.b<-EPS)[0];
        if(negative_ids.shape[0]==0):
            if(self.verb>0):
                print "[%s] INFO Com acceleration is unbounded below"%(self.name);
            return (0, -1e100, 0.0, 1e100);
        a_alpha_d = self.a*alpha-self.d;
        a_alpha_d_negative_bs = a_alpha_d[negative_ids];
        (i_DDalpha_min, DDalpha_min) = [(i,a_min) for (i, a_min) in [(j, a_alpha_d[j]) for j in negative_ids] if (a_min >= a_alpha_d_negative_bs).all()][0];
        ''' Find alpha_max (i.e. value of alpha corresponding to right vertex of active inequality) '''
        den = self.b*self.a[i_DDalpha_min] + self.a;
        i_pos = np.where(den>EPS)[0];
        if(i_pos.shape[0]==0):
            if(self.verb>0):
                print "[%s] INFO b*a_i0+a is never positive, that means that alpha_max is unbounded"%self.name, den;
            alpha_max = 1e100;
        else:
            alpha_max = np.min((self.d[i_pos] + self.b[i_pos]*self.d[i_DDalpha_min])/den[i_pos]);
            if(alpha_max<alpha):
                if(self.verb>0):
                    print "[%s] ERROR: alpha_max<alpha"%self.name, alpha_max, alpha;
                imode = 2;
        return (imode, DDalpha_min, self.a[i_DDalpha_min], alpha_max);
            
    def plotFeasibleAccPolytope(self, range_plot=2.0):
        from pinocchio_inv_dyn.geom_utils import plot_inequalities
        ax = plot_inequalities(np.vstack([self.a,self.b]).T, self.d, [-range_plot,range_plot], [-range_plot,range_plot]);
        plt.axis([0,range_plot,-range_plot,0])
        plt.title('Feasible com pos-acc');
        plut.movePlotSpines(ax, [0, 0]);
        ax.set_xlabel('com pos');
        ax.set_ylabel('com acc');
        plt.show();
        
        
def test(N_CONTACTS = 2, verb=0, solver='qpoases'):
    from math import atan, pi
    np.set_printoptions(precision=2, suppress=True, linewidth=250);
    DO_PLOTS = False;
    PLOT_3D = False;
    mass = 75;             # mass of the robot
    g_vector = np.array([0,0,-9.81]);
    mu = 0.5;           # friction coefficient
    lx = 0.1;           # half foot size in x direction
    ly = 0.07;          # half foot size in y direction
    USE_DIAGONAL_GENERATORS = True;
    GENERATE_QUASI_FLAT_CONTACTS = False;
    #First, generate a contact configuration
    CONTACT_POINT_UPPER_BOUNDS = [ 0.5,  0.5,  0.5];
    CONTACT_POINT_LOWER_BOUNDS = [-0.5, -0.5,  0.0];
    gamma = atan(mu);   # half friction cone angle
    RPY_LOWER_BOUNDS = [-2*gamma, -2*gamma, -pi];
    RPY_UPPER_BOUNDS = [+2*gamma, +2*gamma, +pi];
    MIN_CONTACT_DISTANCE = 0.3;
    
    X_MARG = 0.07;
    Y_MARG = 0.07;
    
    succeeded = False;
    
    while(succeeded == False):
        (p, N) = generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, 
                                   RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, GENERATE_QUASI_FLAT_CONTACTS);        
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
        
    comAccLP = ComAccLP("comAccLP", c0, dc0, p, N, mu, g_vector, mass, verb=verb, regularization=1e-5, solver=solver);
#    comAccLP2 = ComAccLP("comAccLP2", c0, dc0, p, N, mu, g_vector, mass, verb=verb, regularization=1e-5);
    comAccPP = ComAccPP("comAccPP", c0, dc0, p, N, mu, g_vector, mass, verb=verb);
    alpha = 0.0;
    ddAlpha = -1.0;
    while(ddAlpha<0.0):
        if(verb>1):
            print "Compute comAccLP with alpha=%.3f"%alpha;
        (imode, ddAlpha, slope, alpha_min, alpha_max) = comAccLP.compute_max_deceleration(alpha);
#        (imode3, ddAlpha3, slope3, alpha_min3, alpha_max3) = comAccLP2.compute_max_deceleration(alpha);
            
#        if(imode3==0):
#            print "LP2 ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f, qp time %.3f)"%(alpha,ddAlpha3,slope3,alpha_max3,1e3*comAccLP2.qpTime);
#            f = comAccLP2.getContactForces();
#            print "Contact forces:", norm(f, axis=0)
#        else:
#            print "LP failed!"

        if(verb>1):
            print "Compute comAccPP with alpha=%.3f"%alpha;
        (imode2, ddAlpha2, slope2, alpha_max2) = comAccPP.compute_max_deceleration(alpha);            
        
        if(imode!=0 and imode2==0):
            print "ERROR LP failed while PP did not!";
        if(imode==0 and imode2!=0):
            print "ERROR PP failed while LP did not!";
            
        error = False;
        if(imode!= 0 or imode2!=0):
            break;
        if(abs(ddAlpha-ddAlpha2) > 1e-4):
            print "\n *** ERROR ddAlpha", ddAlpha-ddAlpha2;
            error = True;
        if(abs(slope-slope2) > 1e-3):
            print "\n *** ERROR slope", slope-slope2;
            error = True;
        if(abs(alpha_max-alpha_max2) > 1e-3):
            if(alpha_max < alpha_max2):
                print "\n *** WARNING alpha_max underestimated", alpha_max-alpha_max2;
            else:
                print "\n *** ERROR alpha_max", alpha_max-alpha_max2;
                error = True;

        if(error and verb>0):
            print "LP  ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f, qp time %.3f)"%(alpha,ddAlpha,slope,alpha_max,1e3*comAccLP.getLpTime());
            f = comAccLP.getContactForces();
            #print "Contact forces:\n", f;
            print "Contact force norms:", norm(f, axis=0);
            print "Contact force generator coefficients:", comAccLP.x;
            #print "Contact force generators:\n", comAccLP.G4[:,:16];
            print "PP ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f)"%(alpha,ddAlpha2,slope2,alpha_max2);
            print "p:", p.T;
            print "N", N.T;
            print "c", c0.T;
            print "dc", dc0.T;
                        
        alpha = alpha_max+EPS;
        
#    print "\n\n\n SECOND TEST \n\n\n"
#    from com_acc_LP_backup import ComAccLP as ComAccLP_LP
#    alpha = 0.0;
#    ddAlpha = -1.0;
#    while(ddAlpha<0.0):
#        (imode, ddAlpha, slope, alpha_min, alpha_max) = comAccLP2.compute_max_deceleration(alpha);
#        if(imode==0):
#            print "ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f, qp time %.3f)"%(alpha,ddAlpha,slope,alpha_max,1e3*comAccLP2.qpTime);
#        else:
#            print "LP failed!"
#
#        (imode2, ddAlpha2, slope2, alpha_max2) = comAccPP.compute_max_deceleration(alpha);            
#        if(imode2==0):
#            print "ddAlpha_min(%.2f) = %.3f (slope %.3f, alpha_max %.3f)"%(alpha,ddAlpha2,slope2,alpha_max2);
#        else:
#            print "PP failed!";
#            
#        if(imode!= 0 or imode2!=0):
#            break;
#        if(abs(ddAlpha-ddAlpha2) > 1e-4):
#            print "\n *** ERROR ddAlpha", ddAlpha-ddAlpha2;
#        if(abs(slope-slope2) > 1e-3):
#            print "\n *** ERROR slope", slope-slope2;
#        if(abs(alpha_max-alpha_max2) > 1e-3):
#            if(alpha_max < alpha_max2):
#                print "\n *** WARNING alpha_max", alpha_max-alpha_max2;
#            else:
#                print "\n *** ERROR alpha_max", alpha_max-alpha_max2;
#        alpha = alpha_max+EPS;

    if(DO_PLOTS):
        comAccPP.plotFeasibleAccPolytope();
    


if __name__=="__main__":
    import cProfile
    N_CONTACTS = 2;
    SOLVER = 'cvxopt' # qpoases scipy
    VERB = 0;
#    N_TESTS = range(392,393);
    N_TESTS = range(0,100);
    
    for i in N_TESTS:
        try:
            np.random.seed(i);
            print "Test %d"%i
            test(N_CONTACTS, VERB, SOLVER);
#            ret = cProfile.run("test()");
        except Exception as e:
            print e;
            raise
            continue;
