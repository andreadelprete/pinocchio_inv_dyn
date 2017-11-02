import numpy as np
import pinocchio_inv_dyn.optimization.solver_LP_abstract as optim #import optim.getNewSolver, LP_status, LP_status_string
from pinocchio_inv_dyn.multi_contact.utils import compute_centroidal_cone_generators
from pinocchio_inv_dyn.sot_utils import crossMatrix
import time
import random

class EquilibriumExtremumLP (object):
    """
    Compute the extremum CoM position over the line a*x + a0 that is in robust equilibrium.
   * This amounts to solving the following LP:
   *     find          c, b
   *     maximize      c
   *     subject to    A f = D (a c + a0) + d
   *                   f  >= 0
   *   where:
   *     f         are the m coefficients of the contact force generators
   *     c         is the 1d line parameter
   *     A         is the 6xm matrix whose columns are the gravito-inertial wrench generators
   *     D         is the 6x3 matrix mapping the CoM position in gravito-inertial wrench
   *     d         is the 6d vector containing the gravity part of the gravito-inertial wrench
   * @param a 2d vector representing the line direction
   * @param a0 2d vector representing an arbitrary point over the line
   * @return The status of the LP solver.
   * @note If the system is in force closure the status will be unbounded, meaning that the
   * system can reach infinite equilibrium positions. This is due to the fact that we are not considering
   * any upper limit for the friction cones.
    """
        
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of equalities/inequalities
    x = [];    # last solution
    grad = [];     # gradient
    
    solver=None;  # type of solver to use
    maxIter=0;  # max number of iterations
    verb=0;     # verbosity level of the solver (0=min, 2=max)
    
    computationTime = 0.0;  # total computation time
    
    epsilon = np.sqrt(np.finfo(float).eps);

    def __init__ (self, name, contact_points, contact_normals, mu, g, mass, a, a0, 
                  contact_tangents=None, maxIter=100, verb=0, solver='cvxopt'):
        self.name       = name;
        self.solver     = optim.getNewSolver(solver, name, maxIter=maxIter, verb=verb);
        self.maxIter    = maxIter;
        self.verb       = verb;
        self.m_in       = 6;
        self.a          = np.asarray(a).squeeze().copy();
        self.a0         = np.asarray(a0).squeeze().copy();
        
        # compute matrix A, which maps the force generator coefficients into the centroidal wrench
        (self.A, G4) = compute_centroidal_cone_generators(contact_points, contact_normals, mu, contact_tangents);
        self.D = np.zeros((6,3));
        self.d = np.zeros(6);
        self.D[3:,:] = -mass*crossMatrix(g);
        self.d[:3]   = mass*g;
        self.n          = self.A.shape[1]+1;
        
        self.grad = np.zeros(self.n);
        self.grad[0] = -1.0;
        self.constrMat = np.zeros((self.m_in,self.n));
        self.constrMat[:,1:] = self.A;
        self.constrMat[:,0] = -np.dot(self.D, self.a);
        self.constrB = self.d + np.dot(self.D, self.a0);
        self.lb = np.zeros(self.n);
        self.lb[0] = -1e10;
        self.ub = np.array(self.n*[1e10,]);
        self.x = np.zeros(self.n);

    def find_extreme_com_position(self, a=None, a0=None):
        start = time.time();
        if(a is not None and a0 is not None):
            self.a          = np.asarray(a).squeeze().copy();
            self.a0         = np.asarray(a0).squeeze().copy();
            self.constrMat[:,0] = -np.dot(self.D, self.a);
            self.constrB = self.d + np.dot(self.D, self.a0);

        (status, self.x, self.y) = self.solver.solve(self.grad, self.lb, self.ub, A_eq=self.constrMat, b=self.constrB);
        self.computationTime     = time.time()-start;
        
        return (status, self.a0 + self.a*self.x[0]);

    def find_approximate_distance_to_boundaries(self, a0=None, n_directions=8):
        ''' Compute the distance between the specified 3d point a0 and the boundaries of the
            support polygon. Distance is positive if the point is inside the support
            polygon, negative otherwise.
        '''
        start = time.time();
        if(a0 is not None):
            self.a0         = np.asarray(a0).squeeze().copy();
            self.constrB = self.d + np.dot(self.D, self.a0);

        theta = random.uniform(0, 2*np.pi);
        a = np.zeros(3);
        distance = 1e100;
        outside = False;
        status_final = optim.LP_status.INFEASIBLE;
        for i in range(n_directions):
            a[0] = np.cos(theta);
            a[1] = np.sin(theta);
            self.constrMat[:,0] = -np.dot(self.D, a);
            (status, self.x, self.y) = self.solver.solve(self.grad, self.lb, self.ub, A_eq=self.constrMat, b=self.constrB);
            if(status==optim.LP_status.OPTIMAL):
                status_final = status
                if(self.verb>1):
                    print "[%s] Distance to boundaries in direction (%.2f, %.2f) is %.3f"%(self.name, a[0], a[1], self.x[0]);
                if(abs(self.x[0])<distance):
                    if(self.verb>1):
                        print "[%s] Update distance from %.3f to %.3f"%(self.name, distance, self.x[0]);
                    distance = abs(self.x[0]);
                if(self.x[0]<0.0):
                    outside = True;
                    if(self.verb>1):
                        print "[%s] Distance to boundaries in direction (%.2f, %.2f) is negative, hence point is outside"%(self.name, a[0], a[1]);
            elif(self.verb>1):
                print "[%s] Distance to boundaries in direction (%.2f, %.2f) could not be computed: %s"%(self.name, a[0], a[1], optim.LP_status_string[status]);
            theta += 2*np.pi/n_directions;
        
        if(outside):
            distance = -distance;
        self.computationTime     = time.time()-start;
        
        return (status_final, distance);


def test(N_CONTACTS = 2, verb=0, solver='qpoases'):
    from math import atan, pi
    from pinocchio_inv_dyn.multi_contact.utils import generate_contacts, find_static_equilibrium_com, compute_GIWC, compute_support_polygon
    import pinocchio_inv_dyn.plot_utils as plut
    from pinocchio_inv_dyn.geom_utils import plot_inequalities
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=2, suppress=True, linewidth=250);
    DO_PLOTS = True;
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
        
    a = dc0 / np.linalg.norm(dc0);
    equiLP = EquilibriumExtremumLP("quiLP", p, N, mu, g_vector, mass, a, c0, verb=verb, solver=solver);
    (status, com_extr) = equiLP.find_extreme_com_position();
    print "status", optim.LP_status_string[status], "com", com_extr;
    
    if(DO_PLOTS):
        f, ax = plut.create_empty_figure();
        for j in range(p.shape[0]):
            ax.scatter(p[j,0], p[j,1], c='k', s=100);
        ax.scatter(c0[0], c0[1], c='r', s=100);
        com_x = np.zeros(2);
        com_y = np.zeros(2);
        com_x[0] = c0[0]; 
        com_y[0] = c0[1];
        com_x[1] = c0[0]+0.1*dc0[0]; 
        com_y[1] = c0[1]+0.1*dc0[1];
        ax.plot(com_x, com_y, color='b');
        ax.scatter(com_extr[0], com_extr[1], c='r', s=100);
        plt.axis([X_LB,X_UB,Y_LB,Y_UB]);           
        plt.title('Contact Points and CoM position'); 

        (H,h) = compute_GIWC(p, N, mu);
        (B_sp, b_sp) = compute_support_polygon(H, h, mass, g_vector, eliminate_redundancies=False);
        X_MIN = np.min(p[:,0]);
        X_MAX = np.max(p[:,0]);
        X_MIN -= 0.1*(X_MAX-X_MIN);
        X_MAX += 0.1*(X_MAX-X_MIN);
        Y_MIN = np.min(p[:,1]);
        Y_MAX = np.max(p[:,1]);
        Y_MIN -= 0.1*(Y_MAX-Y_MIN);
        Y_MAX += 0.1*(Y_MAX-Y_MIN);
        plot_inequalities(B_sp, b_sp, [X_MIN,X_MAX], [Y_MIN,Y_MAX], ax=ax, color='b', lw=4);
        plt.show();


if __name__=="__main__":
    import cProfile
    N_CONTACTS = 2;
    SOLVER = 'cvxopt' # qpoases scipy
    VERB = 0;
    N_TESTS = range(0,10);
    
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
