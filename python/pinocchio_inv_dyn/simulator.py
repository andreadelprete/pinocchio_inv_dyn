import numpy as np
from numpy.linalg import norm
from numpy.random import random
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
import pinocchio as se3
from staggered_projections import StaggeredProjections
from constraint_violations import ForceConstraintViolation, PositionConstraintViolation, VelocityConstraintViolation, TorqueConstraintViolation, ConstraintViolationType
from first_order_low_pass_filter import FirstOrderLowPassFilter
import time
from pinocchio.explog import exp
from viewer_utils import Viewer


EPS = 1e-4;

def zeros(shape):
    if(isinstance(shape, np.int)):
        return np.matlib.zeros((shape,1));
    elif(len(shape)==2):
        return np.matlib.zeros(shape);
    raise TypeError("The shape is not an int nor a list of two numbers");

    
class Simulator (object):
    name = ''
    q = None;   # current positions
    v = None;   # current velocities
    
    LCP = None;   # approximate LCP solver using staggered projections
    
    USE_LCP_SOLVER          = True;
    DETECT_CONTACT_POINTS   = True;   ''' True: detect collisions between feet and ground, False: collision is specified by the user '''
    GROUND_HEIGHT           = 0.0;
    LOW_PASS_FILTER_INPUT_TORQUES = False;
    
    ENABLE_FORCE_LIMITS = True;
    ENABLE_TORQUE_LIMITS = True;
    ENABLE_JOINT_LIMITS = True;
    
    ACCOUNT_FOR_ROTOR_INERTIAS = False;

    VIEWER_DT = 0.05;
    DISPLAY_COM = True;
    DISPLAY_CAPTURE_POINT = True;
    COM_SPHERE_RADIUS           = 0.01;
    CAPTURE_POINT_SPHERE_RADIUS = 0.01;
    CONTACT_FORCE_ARROW_RADIUS  = 0.01;
    COM_SPHERE_COLOR            = (1, 0, 0, 1); # red, green, blue, alpha
    CAPTURE_POINT_SPHERE_COLOR  = (0, 1, 0, 1);    
    CONTACT_FORCE_ARROW_COLOR   = (1, 0, 0, 1);
    CONTACT_FORCE_ARROW_SCALE   = 1e-3;
    contact_force_arrow_names = [];  # list of names of contact force arrows
    
    SLIP_VEL_THR = 0.1;
    SLIP_ANGVEL_THR = 0.2;
    NORMAL_FORCE_THR = 5.0;
    JOINT_LIMITS_DQ_THR = 1e-1; #1.0;
    TORQUE_VIOLATION_THR = 1.0;
    DQ_MAX = 9.14286;
                       
    ENABLE_WALL_DRILL_CONTACT = False;
    wall_x = 0.5;
    wall_damping = np.array([30, 30, 30, 0.3, 0.3, 0.3]);
    
    k=0;    # number of contact constraints (i.e. size of contact force vector)
    na=0;   # number of actuated DoFs
    nq=0;   # number of position DoFs
    nv=0;   # number of velocity DoFs
    r=[];   # robot
    
    mu=[];          # friction coefficient (force, moment)
    fMin = 0;       # minimum normal force

    dt = 0;     # time step used to compute joint acceleration bounds
    qMin = [];  # joint lower bounds
    qMax = [];  # joint upper bounds
    
    ''' Mapping between y and tau: y = C*tau+c '''
    C = [];
    c = [];
    
    M = [];         # mass matrix
    Md = [];        # rotor inertia
    h = [];         # dynamic drift
    q = [];
    dq = [];

    x_com = [];     # com 3d position
    dx_com = [];    # com 3d velocity
    ddx_com = [];   # com 3d acceleration
    cp = None;      # capture point
#    H_lankle = [];  # left ankle homogeneous matrix
#    H_rankle = [];  # right ankle homogeneous matrix
    J_com = [];     # com Jacobian
    Jc = [];        # contact Jacobian
    
    Minv = [];      # inverse of the mass matrix
    Jc_Minv = [];   # Jc*Minv
    Lambda_c = [];  # task-space mass matrix (Jc*Minv*Jc^T)^-1
    Jc_T_pinv = []; # Lambda_c*Jc_Minv
    Nc_T = [];      # I - Jc^T*Jc_T_pinv
    S_T = [];       # selection matrix
    dJc_v = [];     # product of contact Jacobian time derivative and velocity vector: dJc*v
    
    candidateContactConstraints = [];
    rigidContactConstraints = [];
#    constr_rfoot = None;
#    constr_lfoot = None;
#    constr_rhand = None;
    
#    f_rh = [];      # force right hand
#    w_rf = [];      # wrench right foot
#    w_lf = [];      # wrench left foot
    
    ''' debug variables '''    
    x_c = [];       # contact points
    dx_c = [];      # contact points velocities
    x_c_init = [];  # initial position of constrained bodies    
        
    viewer = None;
    
    def reset(self, t, q, v, dt):
        n = self.nv;
        self.Md = zeros((n,n)); #np.diag([ g*g*i for (i,g) in zip(INERTIA_ROTOR,GEAR_RATIO) ]); # rotor inertia
        self.q  = np.matrix.copy(q);
        self.v = np.matrix.copy(v);
        self.vOld = np.matrix.copy(v);
        self.dv = zeros(n);
        self.dt = dt;
        
        self.S_T        = zeros((n+6,n));
        self.S_T[6:, :] = np.matlib.eye(n);
        self.M          = self.r.mass(self.q);
        self.J_com      = zeros((3,n));

        if(self.DETECT_CONTACT_POINTS==False):
            self.rigidContactConstraints = []; #self.constr_rfoot, self.constr_lfoot];
        else:
            self.rigidContactConstraints = [];
        self.updateInequalityData();

        self.qMin       = self.r.model.lowerPositionLimit;
        self.qMax       = self.r.model.upperPositionLimit;
        self.dqMax      = self.r.model.velocityLimit;
        self.tauMax     = self.r.model.effortLimit;
#        self.ddqMax     = np.array(self.nv*[self.MAX_JOINT_ACC]);
        if(self.freeFlyer):
            self.qMin[:6]   = -1e100;   # set bounds for the floating base
            self.qMax[:6]   = +1e100;
#        self.f_rh        = zeros(6);
#        self.qMin       = np.array(self.r.dynamic.lowerJl.value);
#        self.qMax       = np.array(self.r.dynamic.upperJl.value);
#        self.x_c_init       = zeros(k);
#        self.x_c_init[0:3]  = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];
#        self.x_c_init[6:9]  = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3]
#        self.ddx_c_des = zeros(k);
#        self.computeForwardDynamicMapping(t);
        self.INITIALIZE_TORQUE_FILTER = True;
        
    def __init__(self, name, q, v, fMin, mu, dt, mesh_dir, urdfFileName, freeFlyer=True, detectContactPoint=True):
        self.time_step = 0;
        self.DETECT_CONTACT_POINTS = detectContactPoint;
        self.verb = 0;
        self.name = name;
        self.mu = mu;
        self.fMin = fMin;
        self.freeFlyer = freeFlyer;
        
        if(freeFlyer):
            self.r = RobotWrapper(urdfFileName, mesh_dir, se3.JointModelFreeFlyer());
        else:
            self.r = RobotWrapper(urdfFileName, mesh_dir, None);
        self.nq = self.r.nq;
        self.nv = self.r.nv;
        self.na = self.nv-6 if self.freeFlyer else self.nv;
        
#        self.candidateContactConstraints = [constr_rf_fr, constr_rf_fl, constr_rf_hr, constr_rf_hl,
#                                            constr_lf_fr, constr_lf_fl, constr_lf_hr, constr_lf_hl];
        self.viewer=Viewer(self.name, self.r);
        
        self.reset(0, q, v, dt);
        
        self.LCP = StaggeredProjections(self.nv, self.mu[0], accuracy=EPS, verb=0);
        
        if(self.DISPLAY_COM):
            self.viewer.addSphere('com', self.COM_SPHERE_RADIUS, zeros(3), zeros(3), self.COM_SPHERE_COLOR, 'OFF');
        if(self.DISPLAY_CAPTURE_POINT):
            self.viewer.addSphere('cp', self.CAPTURE_POINT_SPHERE_RADIUS, zeros(3), zeros(3), self.CAPTURE_POINT_SPHERE_COLOR, 'OFF');
        
        
    def updateInequalityData(self):
        c = len(self.rigidContactConstraints);  # number of contacts
        if(self.DETECT_CONTACT_POINTS):
            self.k = c*3;
        else:
            self.k = c*6;   # number of contact force variables
        self.Jc         = zeros((self.k,self.nv));
        self.ddx_c_des  = zeros(self.k);
        self.dJc_v      = zeros(self.k);
        self.C           = np.empty((self.nv+self.k+self.na, self.na));
        self.C[self.nv+self.k:,:]     = np.eye(self.na);
        self.c           = zeros(self.nv+self.k+self.na);
        
    def setTorqueLowPassFilterCutFrequency(self, fc):
        self.LOW_PASS_FILTER_INPUT_TORQUES = True;
        self.TORQUE_LOW_PASS_FILTER_CUT_FREQUENCY = fc;
        self.INITIALIZE_TORQUE_FILTER = True;

    ''' ********** ENABLE OR DISABLE CONTACT CONSTRAINTS ********** '''        

    def removeContactConstraintByName(self, constr_name):
        if(self.DETECT_CONTACT_POINTS==False):
            found = False;
            for i in range(len(self.rigidContactConstraints)):
                if(self.rigidContactConstraints[i].name==constr_name):
                    del self.rigidContactConstraints[i];
                    found = True;
                    break;
            if(found==False):
                print "SIMULATOR: contact constraint %s cannot be removed!" % constr_name;
            self.updateInequalityData();
        
#    def addRightFootContactConstraint(self):
#        self.support_phase=Support.double;
#        
#        if(self.DETECT_CONTACT_POINTS==False):
#            if self.constr_rfoot in self.rigidContactConstraints:
#                return;
#            t = self.rigidContactConstraints[0].opPointModif.position.time;
#            self.constr_rfoot.opPointModif.position.recompute(t+1);
#            self.constr_rfoot.ref = self.constr_rfoot.opPointModif.position.value;
#            self.rigidContactConstraints = self.rigidContactConstraints + [self.constr_rfoot];
#            self.updateInequalityData();
#    
#    def removeRightFootContactConstraint(self):
#        self.support_phase=Support.left;
#        self.removeContactConstraintByName("c_rf_"+self.name);
#        
#    def addLeftFootContactConstraint(self):
#        self.support_phase==Support.double;
#        if(self.DETECT_CONTACT_POINTS==False):
#            if self.constr_lfoot in self.rigidContactConstraints:
#                return;
#            t = self.rigidContactConstraints[0].opPointModif.position.time;
#            self.constr_lfoot.opPointModif.position.recompute(t+1);
#            self.constr_lfoot.ref = self.constr_lfoot.opPointModif.position.value;
#            self.rigidContactConstraints = self.rigidContactConstraints + [self.constr_lfoot];
#            self.updateInequalityData();
#    
#    def removeLeftFootContactConstraint(self):
#        self.support_phase=Support.right;
#        self.removeContactConstraintByName("c_lf_"+self.name);
#        
#    def addRightHandContactConstraint(self):
#        if(self.DETECT_CONTACT_POINTS==False):
#            if self.constr_rhand in self.rigidContactConstraints:
#                return;
#            t = self.rigidContactConstraints[0].opPointModif.position.time;
#            self.constr_rhand.opPointModif.position.recompute(t+1);
#            self.constr_rhand.ref = self.constr_rhand.opPointModif.position.value;
#            self.rigidContactConstraints = self.rigidContactConstraints + [self.constr_rhand];
#            self.updateInequalityData();
#    
#    def removeRightHandContactConstraint(self):
#        self.removeContactConstraintByName("c_rh_"+self.name);


    ''' ********** SET ROBOT STATE ********** '''
    
    def setPositions(self, q, updateConstraintReference=True):
#        for i in range(self.nq):
#            if( q[i]>self.qMax[i]+1e-4 ):
#                print "SIMULATOR Joint %d > upper limit, q-qMax=%f deg" % (i,60*(self.q[i]-self.qMax[i]));
#                q[i] = self.qMax[i]-1e-4;
#            elif( q[i]<self.qMin[i]-1e-4 ):
#                print "SIMULATOR Joint %d < lower limit, qMin-q=%f deg" % (i,60*(self.qMin[i]-self.q[i]));
#                q[i] = self.qMin[i]+1e-4;
        self.q = np.matrix.copy(q);
        self.viewer.updateRobotConfig(q);
        
        if(updateConstraintReference):
            pass;
#            if(self.DETECT_CONTACT_POINTS==False):
#                t = self.rigidContactConstraints[0].opPointModif.position.time;
#                for c in self.rigidContactConstraints:
#                    c.opPointModif.position.recompute(t+1);
#                    c.ref = c.opPointModif.position.value;
#            else:
#                self.rigidContactConstraints = []
#                t = self.candidateContactConstraints[0].opPointModif.position.time;
#                for c in self.candidateContactConstraints:
#                    c.opPointModif.position.recompute(t+1);
#                    if(c.opPointModif.position.value[2][3] < self.GROUND_HEIGHT):
#                        c.ref = c.opPointModif.position.value;
#                        self.rigidContactConstraints = self.rigidContactConstraints + [c,];
#                        print "[SIMULATOR::setPositions] Collision detected for constraint %s" % c.name;
#                        
#            self.x_c_init[0:3]  = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];
#            self.x_c_init[6:9]  = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3];
        return self.q;
    
    def setVelocities(self, v):
        self.v = np.matrix.copy(v);
        return self.v;
        
    
#    def getRightWrist(self):
#        return np.array(self.constr_rhand.opPointModif.position.value);
#        
#    def getRightFootRef(self):
#        return np.array(self.constr_rfoot.ref);
#    
#    def getLeftFootRef(self):
#        return np.array(self.constr_lfoot.ref);
#
#    def getRightFoot(self):
#        return np.array(self.constr_rfoot.opPointModif.position.value);
#        
#    def getLeftFoot(self):
#        return np.array(self.constr_lfoot.opPointModif.position.value);
#        
#    def getRightFootVel(self,t):
#        self.constr_rfoot.task.jacobian.recompute(t);
#        return np.dot(np.array(self.constr_rfoot.task.jacobian.value), self.dq);
#        
#    def getLeftFootVel(self,t):
#        self.constr_lfoot.task.jacobian.recompute(t);
#        return np.dot(np.array(self.constr_lfoot.task.jacobian.value), self.dq);
#        
#    def getRightFootCorners(self,t):
#        pos = zeros(12);
#        vel = zeros(12);
#        acc = zeros(12);
#        i=0;
#        for c in self.candidateContactConstraints:
#            if('c_rf' in c.name):
#                c.task.jacobian.recompute(t);
#                c.task.Jdot.recompute(t);
#                c.opPointModif.position.recompute(t);
#                pos[3*i:3*i+3] = np.array(c.opPointModif.position.value)[:3,3];
#                vel[3*i:3*i+3] = np.dot(np.array(c.task.jacobian.value)[:3,:], self.dq);
#                # acceleration of the previous time step
#                acc[3*i:3*i+3] = np.dot(np.array(c.task.jacobian.value)[:3,:], self.dv) + \
#                                 np.dot(np.array(c.task.Jdot.value)[:3,:], self.dqOld);
#                i += 1;
#        return (pos,vel,acc);
#        
#    def getLeftFootCorners(self,t):
#        pos = zeros(12);
#        vel = zeros(12);
#        acc = zeros(12);
#        i=0;
#        for c in self.candidateContactConstraints:
#            if('c_lf' in c.name):
#                c.task.jacobian.recompute(t);
#                c.task.Jdot.recompute(t);
#                c.opPointModif.position.recompute(t);
#                pos[3*i:3*i+3] = np.array(c.opPointModif.position.value)[:3,3];
#                vel[3*i:3*i+3] = np.dot(np.array(c.task.jacobian.value)[:3,:], self.dq);
#                # acceleration of the previous time step
#                acc[3*i:3*i+3] = np.dot(np.array(c.task.jacobian.value)[:3,:], self.dv) + \
#                                 np.dot(np.array(c.task.Jdot.value)[:3,:], self.dqOld);
#                i += 1;
#        return (pos,vel,acc);
#        
#    def getLeftFootCornersRef(self):
#        pos = zeros(12);
#        i=0;
#        for c in self.candidateContactConstraints:
#            if('c_lf' in c.name):
#                pos[3*i:3*i+3] = np.array(c.ref)[:3,3];
#                i += 1;
#        return pos;
#
#    def getLeftFootCornersAccDes(self):
#        res = zeros(12);
#        i=0;
#        for c in self.candidateContactConstraints:
#            if('c_lf' in c.name):
#                res[3*i:3*i+3] = np.array(c.task.task.value)[0:3];
#                i += 1;
#        return res;
#        
#    def areAllFootCornersInContact(self, name):
#        counter = 0;
#        for c in self.rigidContactConstraints:
#            if(name in c.name):
#                counter += 1;
#        return counter==4;
        
#    def computeForwardDynamicMapping(self, t):
#        n = self.n;
#        self.t = t;
#        
#        self.r.dynamic.com.recompute(t);
#        self.r.dynamic.Jcom.recompute(t);
#        self.r.dynamic.inertia.recompute(t);
##        self.dynamic.dynamicDrift.recompute(t);
#        
#        self.dJ_com   = (np.array(self.r.dynamic.Jcom.value) - self.J_com)/self.dt;
#                
#        i = 0;
#        if(self.DETECT_CONTACT_POINTS==True):
#            ''' COLLISION DETECTION '''
#            constraintsChanged = False;
#            oldConstraints = list(self.rigidContactConstraints); # copy list
#            for c in self.candidateContactConstraints:
#                # do not update task values here because task ref may change in this for loop
#                c.task.jacobian.recompute(t);
#                c.task.Jdot.recompute(t);
#                c.opPointModif.position.recompute(t);
#                if(c in self.rigidContactConstraints):
#                    if(c.opPointModif.position.value[2][3] > self.GROUND_HEIGHT):
#                        j = oldConstraints.index(c);
#                        if(self.f[3*j+2]<EPS):
#                            self.rigidContactConstraints.remove(c);
#                            constraintsChanged = True;
#                        elif(self.verb>0):
#                            print "Collision lost for constraint %s, but I'm gonna keep it because previous normal force was %f" % (c.name, self.f[3*j+2]);                        
#                else:
#                    if(c.opPointModif.position.value[2][3] <= self.GROUND_HEIGHT):
#                        c.ref = c.opPointModif.position.value;
#                        self.rigidContactConstraints.append(c);                        
#                        constraintsChanged = True;
#                        if(self.verb>0):
#                            print "Contact detected for constraint %s, pos %.3f %.3f %.3f" % (c.name
#                                                                                              ,c.opPointModif.position.value[0][3]
#                                                                                              ,c.opPointModif.position.value[1][3]
#                                                                                              ,c.opPointModif.position.value[2][3]);
#            if(constraintsChanged):
#                self.updateInequalityData();
#            for constr in self.rigidContactConstraints:
#                # now update task values
#                constr.task.task.recompute(t);
#                self.Jc[i*3:i*3+3,:]      = np.array(constr.task.jacobian.value)[0:3,:];
#                self.dJc_v[i*3:i*3+3]     = np.dot(constr.task.Jdot.value, self.dq)[0:3];
#                # do not compensate for drift in tangential directions
#                self.ddx_c_des[i*3:i*3+2] = zeros(2);
#                self.ddx_c_des[i*3+2]     = np.array(constr.task.task.value)[2];
#                i = i+1;
#        else:            
#            for constr in self.rigidContactConstraints:
#                constr.task.task.recompute(t);
#                constr.task.jacobian.recompute(t);
#                constr.task.Jdot.recompute(t);
#                self.Jc[i*6:i*6+6,:]      = np.array(constr.task.jacobian.value);
#                self.ddx_c_des[i*6:i*6+6] = np.array(constr.task.task.value); # - np.dot(constr.task.Jdot.value, self.dq);
#                self.dJc_v[i*6:i*6+6]     = np.dot(constr.task.Jdot.value, self.dq);
#                i = i+1;
#
#        self.M        = np.array(self.r.dynamic.inertia.value);
#        if(self.ACCOUNT_FOR_ROTOR_INERTIAS):
#            self.M += self.Md;
#        self.h          = self.M[:,2]*9.81; #np.array(r.dynamic.dynamicDrift.value);
#        self.h          += np.dot(np.array(JOINT_VISCOUS_FRICTION), self.dq);
#        self.dJc_v      -= self.ddx_c_des;
#        self.Minv       = np.linalg.inv(self.M);
#        
#        k = self.k;
#        if(k>0):
#            self.Jc_Minv     = np.dot(self.Jc, self.Minv);
#            self.Lambda_c    = np.linalg.inv(np.dot(self.Jc_Minv, self.Jc.transpose())+0e-10*np.identity(k));
#            self.Jc_T_pinv   = np.dot(self.Lambda_c, self.Jc_Minv);
#            self.Nc_T        = np.eye(n+6) - np.dot(self.Jc.transpose(), self.Jc_T_pinv);
#            self.dx_c        = np.dot(self.Jc, self.dq);
#        
#        ''' WALL CONTACT FORCE COMPUTATION '''
#        self.constr_rhand.opPointModif.position.recompute(t);
#        self.constr_rhand.opPointModif.jacobian.recompute(t);
#        self.R_rh        = np.array(self.constr_rhand.opPointModif.position.value)[0:3,0:3]; # rotation from ref frame to world frame
#        self.x_rh        = np.array(self.constr_rhand.opPointModif.position.value)[0:3,3]; # hand pos w.r.t. world frame expressed in world frame
#        self.J_rh        = np.array(self.constr_rhand.opPointModif.jacobian.value);
#        self.dx_rh       = np.dot(self.J_rh, self.dq);  # hand vel (w.r.t. ref frame) expressed in ref frame
#        self.h_hat = np.matrix.copy(self.h);
#        # conpute wall contact force        
#        self.f_rh       = zeros(6);      # hand force in world frame
#        self.f_rh_local = zeros(6);      # hand force in local frame
#        if(self.ENABLE_WALL_DRILL_CONTACT==True and self.x_rh[0]>=self.wall_x):
#            dx_rh_w = zeros(6);
#            dx_rh_w[:3] = np.dot(self.R_rh, self.dx_rh[:3]);             # hand vel in world frame
#            dx_rh_w[3:] = np.dot(self.R_rh, self.dx_rh[3:]);             # hand vel in world frame
#            self.f_rh = -np.multiply(self.wall_damping, dx_rh_w);     # hand force in world frame
#            if(self.f_rh[0]>0.0):
#                self.f_rh[0] = 0.0; # cannot pull on the wall!
#            self.f_rh_local[:3] = np.dot(self.R_rh.T, self.f_rh[:3]);
#            self.f_rh_local[3:] = np.dot(self.R_rh.T, self.f_rh[3:]);
#            self.h_hat = self.h_hat - np.dot(self.J_rh.transpose(), self.f_rh_local);
#            if(self.verb>0):
#                print "SIM Wall force (local frame)", self.f_rh, 'pos (world frame) %.3f'%self.x_rh[0];
#                print "Hand vel (world frame)", dx_rh_w, 'pos (world frame) %.3f'%self.x_rh[0];
#            
#            
#        ''' Compute C and c such that y = C*tau + c, where y = [dv, f, tau] '''
#        if(self.k>0):
#            self.C[0:n+6,:]      = np.dot(self.Minv, np.dot(self.Nc_T, self.S_T));
#            self.C[n+6:n+6+k,:]  = -np.dot(self.Jc_T_pinv, self.S_T);
#            self.c[0:n+6]        = - np.dot(self.Minv, (np.dot(self.Nc_T,self.h_hat) + np.dot(self.Jc.transpose(), np.dot(self.Lambda_c, self.dJc_v))));
#            self.c[n+6:n+6+k]    = np.dot(self.Lambda_c, (np.dot(self.Jc_Minv, self.h_hat) - self.dJc_v));
#        else:
#            self.C[0:n+6,:]      = np.dot(self.Minv, self.S_T);
#            self.c[0:n+6]        = - np.dot(self.Minv, self.h_hat);
#            
#            
#        
#        '''  Compute capture point '''
#        self.r.dynamic.Jcom.recompute(t);
#        self.r.dynamic.com.recompute(t);
#        self.J_com    = np.array(self.r.dynamic.Jcom.value);
#        self.x_com    = np.array(self.r.dynamic.com.value);
#        self.dx_com     = np.dot(self.J_com, self.dq);
#        self.cp         = self.x_com[0:2] + self.dx_com[0:2]/np.sqrt(9.81/self.x_com[2]);
#                
#        return (self.C, self.c);
#        
#                    
#    ''' Update the state of the robot applying the specified control torques for
#        the specified time step. Return true if everything went fine, false otherwise.
#        Before calling integrate you should call computeForwardDynamicMapping. '''
#    def integrate(self, t, dt, tau):
#        n = self.n;
#        k = self.k;
#        
#        if(self.LOW_PASS_FILTER_INPUT_TORQUES):
#            if(self.INITIALIZE_TORQUE_FILTER):
#                self.torqueFilter = FirstOrderLowPassFilter(self.dt, self.TORQUE_LOW_PASS_FILTER_CUT_FREQUENCY, tau);
#                self.INITIALIZE_TORQUE_FILTER = False;
#            self.tau = self.torqueFilter.filter_data(np.matrix.copy(tau));
#        else:
#            self.tau = tau;
#        
#        self.qOld  = np.matrix.copy(self.q);
#        self.dqOld = np.matrix.copy(self.dq);
#        if(self.USE_LCP_SOLVER):
#            self.Jc_list = len(self.rigidContactConstraints)*[None,];
#            self.dJv_list = len(self.rigidContactConstraints)*[None,];
#            i = 0;
#            for constr in self.rigidContactConstraints:
#                self.Jc_list[i]      = np.array(constr.task.jacobian.value)[0:3,:];
#                self.dJv_list[i]     = np.dot(constr.task.Jdot.value, self.dq)[0:3] - np.array(constr.task.task.value)[0:3];
#                i = i+1;
#                
#            (v, self.f) = self.LCP.simulate(self.dq, self.M, self.h_hat, self.tau, dt, self.Jc_list, self.dJv_list, maxIter=None, maxTime=100.0);
#            self.dv = (v-self.dqOld)/dt;            
#        else:
#            ''' compute accelerations and contact forces from joint torques '''
#            y = np.dot(self.C, self.tau) + self.c;
#            self.dv = y[0:n+6];
#            self.f = y[n+6:n+6+k];
#            
#        return self.integrateAcc(t, dt, self.dv, self.f, self.tau);
    
    def integrateAcc(self, t, dt, dv, f, tau, updateViewer=True):
        res = [];
        self.t = t;
        self.time_step += 1;
        self.dv = np.matrix.copy(dv);
        
        if(abs(norm(self.q[3:7])-1.0) > EPS):
            print "SIMULATOR ERROR Time %.3f "%t, "norm of quaternion is not 1=%f" % norm(self.q[3:7]);
            
        ''' Integrate velocity and acceleration '''
        self.q  = se3.integrate(self.r.model, self.q, dt*self.v);
        self.v += dt*self.dv;
        
        ''' Check for violation of torque limits'''
#        for i in range(self.n):
#            if( tau[i]>TAU_MAX[i]+self.TORQUE_VIOLATION_THR):                    
#                res = res + [TorqueConstraintViolation(self.t*self.dt, i, tau[i])];
#                if(self.ENABLE_TORQUE_LIMITS):
#                    tau[i] = TAU_MAX[i];
#            elif(tau[i]<-TAU_MAX[i]-self.TORQUE_VIOLATION_THR):
#                res = res + [TorqueConstraintViolation(self.t*self.dt, i, tau[i])];
#                if(self.ENABLE_TORQUE_LIMITS):
#                    tau[i] = -TAU_MAX[i];
        
        ''' Compute CoM acceleration '''
#        self.ddx_com = np.dot(self.J_com, self.dv) + np.dot(self.dJ_com, self.v);
        
        ''' DEBUG '''
#        q_pino = sot_2_pinocchio(self.q);
#        v_pino = sot_2_pinocchio_vel(self.dq);
#        g_pino = np.array(self.viewer.robot.gravity(q_pino));
#        h_pino = np.array(self.viewer.robot.biais(q_pino, v_pino));
#        self.viewer.robot.forwardKinematics(q_pino);
#        dJcom_dq = self.viewer.robot.data.oMi[1].rotation*(h_pino[:3] - g_pino[:3]) / self.M[0,0];
#        ddx_com_pino = np.dot(self.J_com, self.dv) + dJcom_dq.reshape(3);
#        
#        self.r.dynamic.position.value = tuple(self.q);
#        self.r.dynamic.velocity.value = tuple(self.dq);
#        self.r.dynamic.com.recompute(self.t+1);
#        self.r.dynamic.Jcom.recompute(self.t+1);
#        new_x_com = np.array(self.r.dynamic.com.value);
#        new_J_com = np.array(self.r.dynamic.Jcom.value);
#        new_dx_com = np.dot(new_J_com, self.dq);
#        new_dx_com_int = self.dx_com + dt*self.ddx_com;
        #if(np.linalg.norm(new_dx_com - new_dx_com_int) > 1e-3):
        #    print "Time %.3f ERROR in integration of com acc"%t, "%.3f"%np.linalg.norm(new_dx_com - new_dx_com_int), new_dx_com, new_dx_com_int;
            
            
        ''' END DEBUG '''
        
#        if(self.DETECT_CONTACT_POINTS==True):
#            ''' compute feet wrenches '''
#            self.w_rf = zeros(6);
#            self.w_lf = zeros(6);
#            X = zeros((6,3));
#            X[:3,:] = np.identity(3);
#            i=0;
#            for c in self.rigidContactConstraints:
#                X[3:,:] = crossMatrix(np.array(c.opmodif)[:3,3] - np.array(H_FOOT_2_SOLE)[:3,3]);
#                if('c_rf' in c.name):
#                    self.w_rf += np.dot(X, self.f[i*3:i*3+3]);
#                elif('c_lf' in c.name):
#                    self.w_lf += np.dot(X, self.f[i*3:i*3+3]);
#                i += 1;
#        else:
#            self.w_rf = f[:6];
#            self.w_lf = f[6:];
#                
#        ''' check for slippage '''
#        if(self.DETECT_CONTACT_POINTS==True):
#            if((self.support_phase==Support.left or self.support_phase==Support.double) and self.areAllFootCornersInContact('c_lf')):
#                v = self.getLeftFootVel(self.t+1);
#                w = self.w_lf;
#                if((np.linalg.norm(v[:3])>self.SLIP_VEL_THR or np.linalg.norm(v[3:])>self.SLIP_ANGVEL_THR) and w[2]>self.NORMAL_FORCE_THR):
#                    res += [ForceConstraintViolation(self.t*self.dt, 'c_lf', w, v)];
#            if((self.support_phase==Support.right or self.support_phase==Support.double) and self.areAllFootCornersInContact('c_rf')):
#                v = self.getRightFootVel(self.t+1);
#                w = self.w_rf;
#                if((np.linalg.norm(v[:3])>self.SLIP_VEL_THR or np.linalg.norm(v[3:])>self.SLIP_ANGVEL_THR) and w[2]>self.NORMAL_FORCE_THR):
#                    res += [ForceConstraintViolation(self.t*self.dt, 'c_rf', w, v)];
#        
#        ''' Check for violation of force limits'''
#        mu = self.mu;
#        if(self.DETECT_CONTACT_POINTS==True):
#            for contactName in ['right_foot','left_foot']:
#                if(contactName=='right_foot'):
#                    fx=self.w_rf[0]; fy=self.w_rf[1]; fz=self.w_rf[2];
#                else:
#                    fx=self.w_lf[0]; fy=self.w_lf[1]; fz=self.w_lf[2];
#                if(fx+mu[0]*fz<-2*EPS or -fx+mu[0]*fz<-2*EPS):
#                    if(fz!=0.0 and self.verb>0):
#                        print "SIMULATOR: friction cone %s x violated, fx=%f, fz=%f, fx/fz=%f" % (contactName,fx,fz,fx/fz);
#                if(fy+mu[0]*fz<-2*EPS or -fy+mu[0]*fz<-2*EPS):
#                    if(fz!=0.0 and self.verb>0):
#                        print "SIMULATOR: friction cone %s y violated, fy=%f, fz=%f, fy/fz=%f" % (contactName,fy,fz,fy/fz);
#                if(fz<-2*EPS and self.verb>0):
#                    print "SIMULATOR: normal force %s z negative, fz=%f" % (contactName,fz);
#        else:                
#            for i in range(len(self.rigidContactConstraints)):
#                if('lf' in self.rigidContactConstraints[i].name):
#                    footSizes = LEFT_FOOT_SIZES;
#                else:
#                    footSizes = RIGHT_FOOT_SIZES
#                fx = f[i*6+0]; fy = f[i*6+1]; fz = f[i*6+2];
#                mx = f[i*6+3]; my = f[i*6+4]; mz = f[i*6+5];
#                # 4 unilateral for linearized friction cone
#                if(fx+mu[0]*fz<-EPS or -fx+mu[0]*fz<-EPS):
#                    if(self.verb>0):
#                        print "SIMULATOR: friction cone x leg %d violated, fx=%f, fz=%f, fx/fz=%f" % (i,fx,fz,fx/fz);
#                    res = res + [ForceConstraintViolation(self.t*self.dt, self.rigidContactConstraints[i].name+' Fx', f[i*6:i*6+6], zeros(6))];
#                if(fy+mu[0]*fz<-EPS or -fy+mu[0]*fz<-EPS):
#                    if(self.verb>0):
#                        print "SIMULATOR: friction cone y leg %d violated, fy=%f, fz=%f, fy/fz=%f" % (i,fy,fz,fy/fz);
#                    res = res + [ForceConstraintViolation(self.t*self.dt, self.rigidContactConstraints[i].name+' Fy', f[i*6:i*6+6], zeros(6))];
#                # 4 unilateral for ZMP
#                if(fz*footSizes[3]-mx<-EPS or fz*footSizes[2]+mx<-EPS):
#                    if(self.verb>0):
#                        print "SIMULATOR: zmp y leg %d violated, mx=%f, fz=%f, zmp=%f, zmpMin=%f, zmpMax=%f" % (
#                            i,mx,fz,mx/fz,footSizes[3],footSizes[2]);
#                    res = res + [ForceConstraintViolation(self.t*self.dt, self.rigidContactConstraints[i].name+' Mx', f[i*6:i*6+6], zeros(6))];
#                if(fz*footSizes[0]+my<-EPS or fz*footSizes[1]-my<-EPS):
#                    if(self.verb>0):
#                        print "SIMULATOR: zmp x leg %d violated, my=%f, fz=%f, zmp=%f, zmpMin=%f, zmpMax=%f" % (
#                            i,my,fz,my/fz,footSizes[0],footSizes[1]);
#                    res = res + [ForceConstraintViolation(self.t*self.dt, self.rigidContactConstraints[i].name+' My', f[i*6:i*6+6], zeros(6))];
#                # 2 unilateral for linearized moment friction cone Mn (normal moment)
##                if(-mz+mu[1]*fz<-EPS or mz+mu[1]*fz<-EPS):
##                    if(self.verb>0):
##                        print "SIMULATOR: friction cone z leg %d violated, mz=%f, fz=%f, mz/fz=%f" % (i,mz,fz,mz/fz);
##                    res = res + [ForceConstraintViolation(self.t*self.dt, self.rigidContactConstraints[i].name+' Mz', f[i*6:i*6+6], zeros(6))];
#                # minimum normal force
#                if(fz<=0.0): #self.fMin-EPS):
#                    if(self.verb>0):
#                        print "SIMULATOR: normal force leg %d violated, fz=%f" % (i,fz);
#                    res = res + [ForceConstraintViolation(self.t*self.dt, self.rigidContactConstraints[i].name+' Fz', f[i*6:i*6+6], zeros(6))];
#                            
        ''' check for violations of joint limits '''
        ind_vel = np.where(np.abs(self.v) > self.DQ_MAX)[0].squeeze();
        ind_vel = np.array([ind_vel]) if len(ind_vel.shape)==0 else ind_vel;
        for i in ind_vel:
            res = res + [VelocityConstraintViolation(self.t*self.dt, i-7, self.v[i], self.dv[i])];
            if(self.verb>0):
                print "[SIMULATOR] %s" % (res[-1].toString());
            if(self.ENABLE_JOINT_LIMITS):
                self.v[i] = self.DQ_MAX if (self.v[i]>0.0) else -self.DQ_MAX;
        
        
        ind_pos_ub = (self.q[7:]>self.qMax[7:]+EPS).A.squeeze();
        ind_pos_lb = (self.q[7:]<self.qMin[7:]-EPS).A.squeeze();
        for i in np.where(ind_pos_ub)[0]:
            res = res + [PositionConstraintViolation(self.t*self.dt, i, self.q[7+i], self.v[6+i], self.dv[6+i])];
            if(self.verb>0):
                print "[SIMULATOR] %s" % (res[-1].toString());
        for i in np.where(ind_pos_lb)[0]:
            res = res + [PositionConstraintViolation(self.t*self.dt, i, self.q[7+i], self.v[6+i], self.dv[6+i])];
            if(self.verb>0):
                print "[SIMULATOR] %s" % (res[-1].toString());
                
        if(self.ENABLE_JOINT_LIMITS):
            self.q[7:][ind_pos_ub] = self.qMax[7:][ind_pos_ub];
            self.v[6:][ind_pos_ub] = 0.0;
            self.q[7:][ind_pos_lb] = self.qMin[7:][ind_pos_lb];
            self.v[6:][ind_pos_lb] = 0.0;
        
        if(updateViewer and self.time_step%int(self.VIEWER_DT/dt)==0):
            self.viewer.updateRobotConfig(self.q);
#            if(self.DISPLAY_COM):
#                self.viewer.updateObjectConfig('com', (self.x_com[0], self.x_com[1], 0, 0,0,0,1));
#            if(self.DISPLAY_CAPTURE_POINT):
#                self.viewer.updateObjectConfig('cp', (self.cp[0], self.cp[1], 0, 0,0,0,1));
            
        return res;
        
    def updateComPositionInViewer(self, com):
        assert np.asarray(com).squeeze().shape[0]==3, "com should be a 3x1 matrix"
        com = np.asarray(com).squeeze();
        if(self.time_step%int(self.VIEWER_DT/self.dt)==0):
            if(self.DISPLAY_COM):
                self.viewer.updateObjectConfig('com', (com[0], com[1], com[2], 0.,0.,0.,1.));
    
    def updateCapturePointPositionInViewer(self, cp):
        assert cp.shape[0]==2, "capture point should be a 2x1 matrix"
        if(self.time_step%int(self.VIEWER_DT/self.dt)==0):
            if(self.DISPLAY_CAPTURE_POINT):
                self.viewer.updateObjectConfig('cp', (cp[0,0], cp[1,0], 0., 0.,0.,0.,1.));
                            
    

    ''' Update the arrows representing the specified contact forces in the viewer.
        If the arrows have not been created yet, it creates them.
        If a force arrow that is currently displayed does not appear in the specified
        list, the arrow visibility is set to OFF.
        @param contact_names A list of contact names
        @param contact_points A list of contact points (i.e. 3x1 numpy matrices)
        @param contact_forces A list of contact forces (i.e. 3x1 numpy matrices)
    '''
    def updateContactForcesInViewer(self, contact_names, contact_points, contact_forces):
        if(self.time_step%int(self.VIEWER_DT/self.dt)==0):
            for (name, p, f) in zip(contact_names, contact_points, contact_forces):
                if(name not in self.contact_force_arrow_names):
                    self.viewer.addArrow(name, self.CONTACT_FORCE_ARROW_RADIUS, p, p+self.CONTACT_FORCE_ARROW_SCALE*f, self.CONTACT_FORCE_ARROW_COLOR);
                    self.viewer.setVisibility(name, "ON");
                    self.contact_force_arrow_names += [name];
                else:
                    self.viewer.moveArrow(name, p, p+self.CONTACT_FORCE_ARROW_SCALE*f);
                    
            for name in self.viewer.arrow_radius:
                if(name not in contact_names):
                    self.viewer.setVisibility(name, "OFF");
                    
            self.contact_force_arrow_names = list(contact_names);