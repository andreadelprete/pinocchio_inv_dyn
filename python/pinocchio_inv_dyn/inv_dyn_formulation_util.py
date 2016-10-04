import numpy as np
from numpy.linalg import norm
from numpy.random import random
from polytope_conversion_utils import cone_span_to_face
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
import pinocchio as se3
from min_jerk_traj_gen import MinimumJerkTrajectoryGenerator
from acc_bounds_util_multi_dof import computeAccLimits
from sot_utils import computeContactInequalities, crossMatrix
from first_order_low_pass_filter import FirstOrderLowPassFilter
from convex_hull_util import compute_convex_hull, plot_convex_hull

EPS = 1e-4;

def zeros(shape):
    if(isinstance(shape, np.int)):
        return np.matlib.zeros((shape,1));
    elif(len(shape)==2):
        return np.matlib.zeros(shape);
    raise TypeError("The shape is not an int nor a list of two numbers");
    
class InvDynFormulation (object):
    name = '';
    verb = 0;
    
    ENABLE_JOINT_LIMITS         = True;
    ENABLE_CAPTURE_POINT_LIMITS = False;
    ENABLE_TORQUE_LIMITS        = True;
    ENABLE_FORCE_LIMITS         = True;
    
    USE_COM_TRAJECTORY_GENERATOR = False;
    USE_JOINT_VELOCITY_ESTIMATOR = False;
    BASE_VEL_FILTER_CUT_FREQ = 5;
    JOINT_VEL_ESTIMATOR_DELAY = 0.02;
    
    ACCOUNT_FOR_ROTOR_INERTIAS = True;
    
    JOINT_FRICTION_COMPENSATION_PERCENTAGE = 1.0;
    MAX_JOINT_ACC = 100.0;      # maximum acceleration upper bound
    MAX_MIN_JOINT_ACC = 10.0;   # maximum acceleration lower bound
    JOINT_POS_PREVIEW = 1; # preview window to convert joint pos limits into joint acc limits
    JOINT_VEL_PREVIEW = 1;  # preview window to convert joint vel limits into joint acc limits
    
    na=0;    # number of actuated joints
    nq=0;    # number of position DoFs
    nv=0;   # number of velocity DoFs
    m_in=0; # number of inequalities
    k=0;    # number of contact constraints (i.e. size of contact force vector)
    
    ind_force_in = [];  # indeces of force inequalities
    ind_acc_in = [];    # indeces of acceleration inequalities
#    ind_torque_in = []; # indeces of torque inequalities
    ind_cp_in = [];     # indeces of capture point inequalities
    
    tauMax=[];  # torque limits

    dt = 0;     # time step used to compute joint acceleration bounds
    qMin = [];  # joint lower bounds
    qMax = [];  # joint upper bounds
    dqMax = []; # max joint velocities
    ddqMax = []; # max joint accelerations
    
    ddqMaxFinal = [];   # max joint accelerations resulting from pos/vel/acc limits
    ddqMinFinal = [];   # min joint accelerations resulting from pos/vel/acc limits

    ''' Classic inverse dynamics formulation
            minimize    ||A*y-a||^2
            subject to  B*y+b >= 0
                        dynamics(y) = 0
        where y=[dv, f, tau]
    '''
    A = [];
    a = [];
    B = [];
    b = [];
    
    ''' Mapping between y and tau: y = C*tau+c '''
    C = [];
    c = [];

    ''' Reformulation of the inverse dynamics optimization problem
        in terms of tau only:
            minimize    ||D*tau-d||^2
            subject to  G*tau+g >= 0
    '''
    D = [];
    d = [];
    G = [];
    g = [];
    
    M = [];         # mass matrix
    h = [];         # dynamic drift
    q = [];
    v = [];
    
    x_com = [];     # com 3d position
    dx_com = [];    # com 3d velocity
    ddx_com = [];   # com 3d acceleration
    cp = None;      # capture point

    mass = 0;
    J_com = [];     # com Jacobian
    Jc = [];        # contact Jacobian
    x_c = [];       # contact points
    dx_c = [];      # contact points velocities
    
    Minv = [];      # inverse of the mass matrix
    Jc_Minv = [];   # Jc*Minv
    Lambda_c = [];  # task-space mass matrix (Jc*Minv*Jc^T)^-1
    Jc_T_pinv = []; # Lambda_c*Jc_Minv
    Nc_T = [];      # I - Jc^T*Jc_T_pinv
    S_T = [];       # selection matrix
    dJc_v = [];     # product of contact Jacobian time derivative and velocity vector: dJc*v
    
    rigidContactConstraints = [];   # tasks associated to the contact constraints
    rigidContactConstraints_p = []; # contact points
    rigidContactConstraints_N = []; # contact normals
    rigidContactConstraints_fMin = [];  # minimum normal forces
    rigidContactConstraints_mu = [];    # friction coefficients
    rigidContactConstraints_m_in = [];  # number of inequalities
    bilateralContactConstraints = [];

    tasks = [];
    task_weights = [];
    
    B_conv_hull = None;     # 2d convex hull of contact points: B_conv_hull*x + b_conv_hull >= 0
    b_conv_hull = None;

    
    def updateInequalityData(self):
        self.m_in = 0;                              # number of inequalities
        c = len(self.rigidContactConstraints);      # number of unilateral contacts
        self.k = int(np.sum([con.dim for con in self.rigidContactConstraints]));
        self.k += int(np.sum([con.dim for con in self.bilateralContactConstraints]));
        if(self.ENABLE_FORCE_LIMITS):
            self.rigidContactConstraints_m_in = np.zeros(c, np.int);
            Bf = zeros((0,self.k));
            bf = zeros(0);
            ii = 0;
            for i in range(c):
                (Bfi, bfi) = self.createContactForceInequalities(self.rigidContactConstraints_fMin[i], self.rigidContactConstraints_mu[i], \
                                                               self.rigidContactConstraints_p[i], self.rigidContactConstraints_N[i]);
                self.rigidContactConstraints_m_in[i] = Bfi.shape[0];
                tmp = zeros((Bfi.shape[0], self.k));
                dim = self.rigidContactConstraints[i].dim;
                mask = self.rigidContactConstraints[i]._mask;
                tmp[:,ii:ii+dim] = Bfi[:,mask];
                ii += dim;
                Bf = np.vstack((Bf, tmp));
                bf = np.vstack((bf, bfi));
            self.ind_force_in = range(self.m_in, self.m_in + np.sum(self.rigidContactConstraints_m_in));
            self.m_in += np.sum(self.rigidContactConstraints_m_in);
        else:
            self.ind_force_in = [];

        if(self.ENABLE_JOINT_LIMITS):
            self.ind_acc_in = range(self.m_in, self.m_in+2*self.na);
            self.m_in += 2*self.na;
        else:
            self.ind_acc_in = [];
            
        if(self.ENABLE_TORQUE_LIMITS):
            self.lb = -self.tauMax;
            self.ub = self.tauMax;
        else:
            self.lb = zeros(self.na) - 1e100;
            self.ub = zeros(self.na) + 1e100;
            
        if(self.ENABLE_CAPTURE_POINT_LIMITS):
            self.ind_cp_in = range(self.m_in, self.m_in+self.b_conv_hull.size);
            self.m_in += self.b_conv_hull.size;
        else:
            self.ind_cp_in = [];
            
        self.B          = zeros((self.m_in, self.nv+self.k+self.na));
        self.b          = zeros(self.m_in);
        self.Jc         = zeros((self.k,self.nv));
        self.dJc_v      = zeros(self.k);
        self.dx_c       = zeros(self.k);
        self.ddx_c_des  = zeros(self.k);
        self.Jc_Minv    = zeros((self.k,self.nv));
        self.Lambda_c   = zeros((self.k,self.k));
        self.Jc_T_pinv  = zeros((self.k,self.nv));
        self.Nc_T       = np.matlib.eye(self.nv);
        self.C           = zeros((self.nv+self.k+self.na, self.na));
        self.c           = zeros(self.nv+self.k+self.na);
        
        if(self.ENABLE_FORCE_LIMITS and self.k>0):
            self.B[self.ind_force_in, self.nv:self.nv+self.k] = Bf;
            self.b[self.ind_force_in] = bf;
#            print "Contact inequality constraints:\n", self.B[self.ind_force_in, self.nv:self.nv+self.k], "\n", bf.T;
        
    
    def __init__(self, name, q, v, dt, mesh_dir, urdfFileName, freeFlyer=True):
        if(freeFlyer):
            self.r = RobotWrapper(urdfFileName, mesh_dir, root_joint=se3.JointModelFreeFlyer());
        else:
            self.r = RobotWrapper(urdfFileName, mesh_dir, None);        
        self.freeFlyer = freeFlyer;
        self.nq = self.r.nq;
        self.nv = self.r.nv;
        self.na = self.nv-6 if self.freeFlyer else self.nv;
        self.k = 0;        # number of constraints
        self.dt = dt;
        self.name = name;
        self.Md = zeros((self.na,self.na)); #np.diag([ g*g*i for (i,g) in zip(INERTIA_ROTOR,GEAR_RATIO) ]); # rotor inertia
        
        ''' create low-pass filter for base velocities '''
        self.baseVelocityFilter = FirstOrderLowPassFilter(dt, self.BASE_VEL_FILTER_CUT_FREQ , np.zeros(6));            
        if(freeFlyer):
            self.S_T         = zeros((self.nv,self.na));
            self.S_T[6:, :]  = np.matlib.eye(self.na);
        else:
            self.S_T    = np.matlib.eye(self.na);
    
        self.qMin       = self.r.model.lowerPositionLimit;
        self.qMax       = self.r.model.upperPositionLimit;
        self.dqMax      = self.r.model.velocityLimit;
        self.ddqMax     = np.matlib.zeros((self.na,1)); 
        self.ddqStop    = np.matlib.zeros((self.na,1));
        if(self.freeFlyer):
            self.qMin[:6]   = -1e100;   # set bounds for the floating base
            self.qMax[:6]   = +1e100;
            self.tauMax     = self.r.model.effortLimit[6:];
        else:
            self.tauMax     = self.r.model.effortLimit;
                        
        self.contact_points = zeros((0,3));
        self.updateConvexHull();
        self.updateInequalityData();
        self.setNewSensorData(0, q, v)
        
        self.com_traj_gen = MinimumJerkTrajectoryGenerator(self.dt, 1);
        
    def getFrameId(self, frameName):
        if(self.r.model.existFrame(frameName)==False):
            raise NameError("[InvDynFormUtil] ERROR: frame %s does not exist!"%frameName);
        return self.r.model.getFrameId(frameName);

    ''' ********** ENABLE OR DISABLE CONTACT CONSTRAINTS ********** '''

    def removeUnilateralContactConstraint(self, constr_name):
        found = False;
        for i in range(len(self.rigidContactConstraints)):
            if(self.rigidContactConstraints[i].name==constr_name):
                del self.rigidContactConstraints[i];
                del self.rigidContactConstraints_p[i];
                del self.rigidContactConstraints_N[i];
                del self.rigidContactConstraints_fMin[i];
                del self.rigidContactConstraints_mu[i];
                found = True;
                break;
        if(found==False):
            for i in range(len(self.bilateralContactConstraints)):
                if(self.bilateralContactConstraints[i].name==constr_name):
                    del self.bilateralContactConstraints[i];
                    found=True;
                    break;
            if(found==False):
                raise ValueError("[InvDynForm] ERROR: constraint %s cannot be removed because it does not exist!" % constr_name);
        self.updateInequalityData();
        
        
    def addUnilateralContactConstraint(self, constr, contact_points, contact_normals, fMin, mu):
        self.rigidContactConstraints        += [constr];
        self.rigidContactConstraints_p      += [contact_points];
        self.rigidContactConstraints_N      += [contact_normals];
        self.rigidContactConstraints_fMin   += [fMin];
        self.rigidContactConstraints_mu     += [mu];
        self.updateInequalityData();
        
    def existUnilateralContactConstraint(self, constr_name):
        res = [c.name for c in self.rigidContactConstraints if c.name==constr_name];
        return True if len(res)>0 else False;
        
    def addTask(self, task, weight):
        self.tasks        += [task];
        self.task_weights += [weight];
        
    def removeTask(self, task_name):
        for (i,t) in enumerate(self.tasks):
            if t.name==task_name:
                del self.tasks[i];
                del self.task_weights[i];
                return True;
        raise ValueError("[InvDynForm] ERROR: task %s cannot be removed because it does not exist!" % task_name);
        
        
    def updateConvexHull(self):
        pass;
#        ''' x_*foot is used for computing the capture-point constraints '''
#        self.x_lfoot = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];
#        self.x_rfoot = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3];
#        
#        ''' Compute positions of foot corners in world frame '''
#        self.x_rf = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3];  # position right foot
#        self.x_lf = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];  # position left foot
#        self.R_rf = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,0:3];  # rotation matrix right foot
#        self.R_lf = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,0:3];  # rotation matrix left foot
#        
#        self.contact_points[0,:] = np.dot(self.R_rf, np.array([ RIGHT_FOOT_SIZES[0],  RIGHT_FOOT_SIZES[2], 0])) + self.x_rf;
#        self.contact_points[1,:] = np.dot(self.R_rf, np.array([ RIGHT_FOOT_SIZES[0], -RIGHT_FOOT_SIZES[3], 0])) + self.x_rf;
#        self.contact_points[2,:] = np.dot(self.R_rf, np.array([-RIGHT_FOOT_SIZES[1],  RIGHT_FOOT_SIZES[2], 0])) + self.x_rf;
#        self.contact_points[3,:] = np.dot(self.R_rf, np.array([-RIGHT_FOOT_SIZES[1], -RIGHT_FOOT_SIZES[3], 0])) + self.x_rf;
#        
#        self.contact_points[4,:] = np.dot(self.R_lf, np.array([ LEFT_FOOT_SIZES[0],  LEFT_FOOT_SIZES[2], 0])) + self.x_lf;
#        self.contact_points[5,:] = np.dot(self.R_lf, np.array([ LEFT_FOOT_SIZES[0], -LEFT_FOOT_SIZES[3], 0])) + self.x_lf;
#        self.contact_points[6,:] = np.dot(self.R_lf, np.array([-LEFT_FOOT_SIZES[1],  LEFT_FOOT_SIZES[2], 0])) + self.x_lf;
#        self.contact_points[7,:] = np.dot(self.R_lf, np.array([-LEFT_FOOT_SIZES[1], -LEFT_FOOT_SIZES[3], 0])) + self.x_lf;
#
#        ''' compute convex hull of foot corners '''
#        (self.B_conv_hull, self.b_conv_hull) = compute_convex_hull(self.contact_points[:,0:2].T);
#        # normalize inequalities
#        for i in range(self.B_conv_hull.shape[0]):
#            tmp = np.linalg.norm(self.B_conv_hull[i,:]);
#            if(tmp>1e-6):
#                self.B_conv_hull[i,:] /= tmp;
#                self.b_conv_hull[i]   /= tmp;
#plot_convex_hull(self.B_conv_hull, self.b_conv_hull, self.contact_points[:,0:2]);
        
        
    ''' ********** ENABLE OR DISABLE INEQUALITY CONSTRAINTS ********** '''
    def enableJointLimits(self, enable=True, IMPOSE_POSITION_BOUNDS=True, IMPOSE_VELOCITY_BOUNDS=True, 
                          IMPOSE_VIABILITY_BOUNDS=True, IMPOSE_ACCELERATION_BOUNDS=True):
        self.ENABLE_JOINT_LIMITS = enable;
        self.IMPOSE_POSITION_BOUNDS = IMPOSE_POSITION_BOUNDS;
        self.IMPOSE_VELOCITY_BOUNDS = IMPOSE_VELOCITY_BOUNDS;
        self.IMPOSE_VIABILITY_BOUNDS = IMPOSE_VIABILITY_BOUNDS;
        self.IMPOSE_ACCELERATION_BOUNDS = IMPOSE_ACCELERATION_BOUNDS;
        self.updateInequalityData();
        
    def enableTorqueLimits(self, enable=True):
        self.ENABLE_TORQUE_LIMITS = enable;
        self.updateInequalityData();
        
    def enableForceLimits(self, enable=True):
        self.ENABLE_FORCE_LIMITS = enable;
        self.updateInequalityData();
        
    def enableCapturePointLimits(self, enable=True):
        self.ENABLE_CAPTURE_POINT_LIMITS = enable;
        self.updateInequalityData();
    
    ''' ********** SET ROBOT STATE ********** '''
    def setPositions(self, q, updateConstraintReference=True):
        self.q = np.matrix.copy(q);
        
        if(updateConstraintReference):
            if(self.USE_JOINT_VELOCITY_ESTIMATOR):
                self.estimator.init(self.dt,self.JOINT_VEL_ESTIMATOR_DELAY,self.JOINT_VEL_ESTIMATOR_DELAY,self.JOINT_VEL_ESTIMATOR_DELAY,self.JOINT_VEL_ESTIMATOR_DELAY,True);
                self.baseVelocityFilter = FirstOrderLowPassFilter(self.dt, self.BASE_VEL_FILTER_CUT_FREQ , np.zeros(6));
            self.r.forwardKinematics(q);
            for c in self.rigidContactConstraints:
                Mref = self.r.position(q, c._link_id, update_geometry=False);
                c.refTrajectory.setReference(Mref);
#                dx = np.dot(c.task.jacobian.value, self.dq);
#                if(np.linalg.norm(dx)>EPS):
#                    print "[InvDynForm] Contact constraint velocity: %.3f" % np.linalg.norm(dx);
            for c in self.bilateralContactConstraints:
                Mref = self.r.position(q, c._link_id, update_geometry=False);
                c.refTrajectory.setReference(Mref);
            self.updateConvexHull();
            
        return self.q;
    
    def setVelocities(self, v):
        if(self.USE_JOINT_VELOCITY_ESTIMATOR):
            raise Exception("Joint velocity estimator not implemented yet");
        else:
            self.v = np.matrix.copy(v);
        return self.v;
        
    def setNewSensorData(self, t, q, v):
        k = self.k;
        nv = self.nv;
        
        self.setPositions(q, updateConstraintReference=False);
        self.setVelocities(v);
        
        self.r.computeAllTerms(q, v);
        self.r.framesKinematics(q);
        self.x_com    = self.r.com(q, update_kinematics=False);
        self.J_com    = self.r.Jcom(q, update_kinematics=False);
        self.M        = self.r.mass(q, update_kinematics=False);
        if(self.ACCOUNT_FOR_ROTOR_INERTIAS):
            if(self.freeFlyer):
                self.M[6:,6:]   += self.Md;
            else:
                self.M   += self.Md;
        self.h        = self.r.bias(q,v, update_kinematics=False);
#        self.h          += self.JOINT_FRICTION_COMPENSATION_PERCENTAGE*np.dot(np.array(JOINT_VISCOUS_FRICTION), self.v);
        self.dx_com     = np.dot(self.J_com, self.v);
        com_z           = self.x_com[2]; #-np.mean(self.contact_points[:,2]);
        self.cp         = self.x_com[:2] + self.dx_com[:2]/np.sqrt(9.81/com_z);

        i = 0;
        for constr in self.rigidContactConstraints:
            dim = constr.dim
            (self.Jc[i:i+dim,:], self.dJc_v[i:i+dim], self.ddx_c_des[i:i+dim]) = constr.dyn_value(t, q, v, local_frame=False);
            i += dim;
        for constr in self.bilateralContactConstraints:
            dim = constr.dim
            (self.Jc[i:i+dim,:], self.dJc_v[i:i+dim], self.ddx_c_des[i:i+dim]) = constr.dyn_value(t, q, v, local_frame=False);
            i += dim;
        self.Minv        = np.linalg.inv(self.M);
        if(self.k>0):
            self.Jc_Minv     = np.dot(self.Jc, self.Minv);
            self.Lambda_c    = np.linalg.inv(np.dot(self.Jc_Minv, self.Jc.T) + 1e-10*np.matlib.eye(self.k));
            self.Jc_T_pinv   = np.dot(self.Lambda_c, self.Jc_Minv);
            self.Nc_T        = np.matlib.eye(self.nv) - np.dot(self.Jc.T, self.Jc_T_pinv);
            self.dx_c        = np.dot(self.Jc, self.v);
        else:
            self.Nc_T        = np.matlib.eye(self.nv);
        
        # Compute C and c such that y = C*tau + c, where y = [dv, f, tau]
        self.C[0:nv,:]      = np.dot(self.Minv, np.dot(self.Nc_T, self.S_T));
        self.C[nv:nv+k,:]   = -np.dot(self.Jc_T_pinv, self.S_T);
        self.C[nv+k:,:]     = np.matlib.eye(self.na);
        self.c[0:nv]        = - np.dot(self.Minv, (np.dot(self.Nc_T,self.h) + np.dot(self.Jc.T, np.dot(self.Lambda_c, self.dJc_v - self.ddx_c_des))));
        self.c[nv:nv+k]     = np.dot(self.Lambda_c, (np.dot(self.Jc_Minv, self.h) - self.dJc_v + self.ddx_c_des));

        
    def computeCostFunction(self, t):
        n_tasks = len(self.tasks);
        dims    = np.empty(n_tasks, np.int);
        J       = n_tasks*[None,];
        drift   = n_tasks*[None,];
        a_des   = n_tasks*[None,];
        dim = 0;
        for k in range(n_tasks):
            J[k], drift[k], a_des[k] = self.tasks[k].dyn_value(t, self.q, self.v);
            dims[k] = a_des[k].shape[0];
            dim += dims[k];
        A = zeros((dim, self.nv+self.k+self.na));
        a = zeros(dim);
        i = 0;
        for k in range(n_tasks):
            A[i:i+dims[k],:self.nv] = self.task_weights[k]*J[k];
            a[i:i+dims[k]]          = self.task_weights[k]*(a_des[k] - drift[k]);
            i += dims[k];
        D       = np.dot(A,self.C);
        d       = a - np.dot(A,self.c);
        return (D,d);
    
    
    ''' ********** GET ROBOT STATE ********** '''        
    def getAngularMomentum(self):
        I = self.M[3:6,3:6];
        return np.dot(np.linalg.inv(I), np.dot(self.M[3:6,:], self.v));
        
    def getZmp(self, f_l, f_r):
        return zeros(2);
#        self.x_rf = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3];  # position right foot
#        self.x_lf = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];  # position left foot
#        self.R_rf = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,0:3];  # rotation matrix right foot
#        self.R_lf = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,0:3];  # rotation matrix left foot
#        
#        self.zmp_l = zeros(3);
#        self.zmp_r = zeros(3);
#        if(abs(f_l[2])>1e-6):
#            self.zmp_l[0] = -f_l[4]/f_l[2];
#            self.zmp_l[1] = f_l[3]/f_l[2];
#        self.zmp_l = self.x_lf + np.dot(self.R_lf, self.zmp_l);
#        if(abs(f_r[2])>1e-6):
#            self.zmp_r[0] = -f_r[4]/f_r[2];
#            self.zmp_r[1] = f_r[3]/f_r[2];
#        self.zmp_r = self.x_rf + np.dot(self.R_rf, self.zmp_r);
#        self.zmp = (f_l[2]*self.zmp_l[:2] + f_r[2]*self.zmp_r[:2]) / (f_l[2]+f_r[2]);
#        return np.matrix.copy(self.zmp);

    
   
    ''' ********** CREATE INEQUALITY CONSTRAINTS ********** '''

    ''' Computes a matrix B and a vector b such that the inequalities:
            B*dv + b >= 0
        ensures that the capture point at the next time step will lie
        inside the support polygon. Note that the vector dv contains the
        accelerations of base+joints of the robot. This methods assumes that
        x_com, dx_com, J_com, B_conv_hull and b_conv_hull have been already computed.
    '''
    def createCapturePointInequalities(self, footSizes = None):    
        dt      = self.dt;
        omega   = np.sqrt(9.81/self.x_com[2]);
        x_com   = self.x_com[0:2];  # only x and y coordinates
        dx_com  = self.dx_com[0:2];
    
        B    = (0.5*dt*dt + dt/omega)*np.dot(self.B_conv_hull, self.J_com[0:2,:]);
        b    = self.b_conv_hull + np.dot(self.B_conv_hull, x_com + (dt+1/omega)*dx_com);
        return (B,b);
        
    def createTorqueInequalities(self, tauMax=None):
        if(tauMax==None):
            tauMax = self.tauMax;
        n = self.na;
        B = zeros((2*n,n));
        b = zeros(2*n);
        B[0:n,:]    =  np.matlib.identity(n);
        B[n:2*n,:]  = -np.matlib.identity(n);
        b[0:n]      = self.tauMax;
        b[n:2*n]    = self.tauMax;
        return (B,b);

    def createJointAccInequalitiesViability(self):
        n  = self.na;
        B  = zeros((2*n,n));
        b  = zeros(2*n);
                
        B[:n,:]  =  np.matlib.identity(n);
        B[n:,:]  = -np.matlib.identity(n);

        # Take the most conservative limit for each joint
        dt = max(self.JOINT_POS_PREVIEW,self.JOINT_VEL_PREVIEW)*self.dt;
        self.ddqMax[:,0]  = self.MAX_JOINT_ACC;
        self.ddqStop[:,0] = self.MAX_MIN_JOINT_ACC;
        (ddqLB, ddqUB) = computeAccLimits(self.q[7:], self.v[6:], self.qMin[7:], self.qMax[7:], self.dqMax[6:], self.ddqMax, 
                                          dt, False, self.ddqStop, self.IMPOSE_POSITION_BOUNDS, self.IMPOSE_VELOCITY_BOUNDS, 
                                          self.IMPOSE_VIABILITY_BOUNDS, self.IMPOSE_ACCELERATION_BOUNDS);
        self.ddqMinFinal = ddqLB;
        self.ddqMaxFinal = ddqUB;
        
        b[:n]    = -self.ddqMinFinal;
        b[n:]    = self.ddqMaxFinal;

        if(np.isnan(b).any()):
            print " ****** ERROR ***** Joint acceleration limits contain nan";
        
        return (B,b);
    
    
    def createContactForceInequalities(self, fMin, mu, contact_points, contact_normals):
        if(contact_points.shape[1]>1):
            B = -1*computeContactInequalities(contact_points.T, contact_normals.T, mu[0]);
            b = zeros(B.shape[0]);
        elif(norm(contact_points)<EPS):
            B = zeros((5,6));
            b = zeros(B.shape[0]);
            B[0,0]   = -1;
            B[1,0]   = 1;
            B[2,1]   = -1;
            B[3,1]   = 1;            
            B[:,2]   = mu[0];
            # minimum normal force
            B[-1,2] = 1;
            b[-1]   = -fMin;
        else:
            raise ValueError("Contact with only one point that is not at the origin of the frame: NOT SUPPORTED");

        return (B,b);
        
    ''' Compute the matrix A and the vectors lbA, ubA such that:
            lbA <= A*tau <= ubA
        ensures that all the inequality constraints the system is subject to are satisfied.
        Before calling this method you should call setNewSensorData to set the current state of 
        the robot.
    '''
    def createInequalityConstraints(self):
        n = self.na;
        k = self.k;

        if(self.ENABLE_JOINT_LIMITS):
            (B_q, b_q) = self.createJointAccInequalitiesViability();
            self.B[self.ind_acc_in, 6:n+6]      = B_q;
            self.b[self.ind_acc_in]             = b_q;
            
        if(self.ENABLE_CAPTURE_POINT_LIMITS):
            (B_cp, b_cp) = self.createCapturePointInequalities();
            self.B[self.ind_cp_in, :n+6]        = B_cp;
            self.b[self.ind_cp_in]              = b_cp;
        
        self.G       = np.dot(self.B, self.C);
        self.glb     = self.b + np.dot(self.B, self.c);
        self.gub     = 1e10*np.matlib.ones((self.m_in,1))
        return (self.G, -self.glb, self.gub, self.lb, self.ub);
    
        
    def createForceRegularizationTask(self, w_f):
        n = self.n;      # number of joints
        k = self.k;
        A = zeros((12,2*n+6+k));
        A[:,n+6:n+6+12]  = np.diag(w_f);
        D       = np.dot(A,self.C);
        d       = - np.dot(A,self.c);
        return (D,d);
