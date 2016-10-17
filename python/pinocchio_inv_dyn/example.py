'''
Simple example of how to use this simulation/control environment
'''

import example_config as conf

from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
from pinocchio_inv_dyn.standard_qp_solver import StandardQpSolver
from pinocchio_inv_dyn.simulator import Simulator
import pinocchio_inv_dyn.viewer_utils as viewer_utils
from pinocchio_inv_dyn.inv_dyn_formulation_util import InvDynFormulation
import pinocchio_inv_dyn.plot_utils as plot_utils
from pinocchio_inv_dyn.tasks import SE3Task, CoMTask, JointPostureTask
from pinocchio_inv_dyn.trajectories import ConstantSE3Trajectory, ConstantNdTrajectory

import pinocchio as se3
from pinocchio.utils import zero as mat_zeros
import cProfile
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep

EPS = 1e-4;

def createListOfMatrices(listSize, matrixSize):
    l = listSize*[None,];
    for i in range(listSize):
        l[i] = np.matlib.zeros(matrixSize);
    return l;

def createListOfLists(size1, size2):
    l = size1*[None,];
    for i in range(size1):
        l[i] = size2*[None,];
    return l;
    

def createInvDynFormUtil(q, v):
    invDynForm = InvDynFormulation('hrp2_inv_dyn', q, v, conf.dt, conf.model_path, conf.urdfFileName, conf.freeFlyer);
    invDynForm.USE_COM_TRAJECTORY_GENERATOR = False;
    invDynForm.enableCapturePointLimits(conf.ENABLE_CAPTURE_POINT_LIMITS);
    invDynForm.enableTorqueLimits(conf.ENABLE_TORQUE_LIMITS);
    invDynForm.enableForceLimits(conf.ENABLE_FORCE_LIMITS);
    invDynForm.enableJointLimits(conf.ENABLE_JOINT_LIMITS, conf.IMPOSE_POSITION_BOUNDS, conf.IMPOSE_VELOCITY_BOUNDS, 
                                 conf.IMPOSE_VIABILITY_BOUNDS, conf.IMPOSE_ACCELERATION_BOUNDS);
    invDynForm.JOINT_POS_PREVIEW            = conf.JOINT_POS_PREVIEW;
    invDynForm.JOINT_VEL_PREVIEW            = conf.JOINT_VEL_PREVIEW;
    invDynForm.MAX_JOINT_ACC                = conf.MAX_JOINT_ACC;
    invDynForm.MAX_MIN_JOINT_ACC            = conf.MAX_MIN_JOINT_ACC;
    invDynForm.USE_JOINT_VELOCITY_ESTIMATOR = conf.USE_JOINT_VELOCITY_ESTIMATOR;
    invDynForm.ACCOUNT_FOR_ROTOR_INERTIAS   = conf.ACCOUNT_FOR_ROTOR_INERTIAS;
    #print invDynForm.r.model
    
    return invDynForm;
    
    
def updateConstraints(t, i, q, v, invDynForm, contacts):
    contact_changed = False;
    
    for active_constr in invDynForm.rigidContactConstraints:
        if(active_constr.name not in contacts.keys()):
            invDynForm.removeUnilateralContactConstraint(active_constr.name);
            print "Time %.3f Removing constraint and adding task"%t, active_constr.name;
            contact_changed =True;

    for name in contacts:
        if(invDynForm.existUnilateralContactConstraint(name)):
            continue;
        
        contact_changed =True;
        invDynForm.r.forwardKinematics(q, v, 0 * v);
        invDynForm.r.framesKinematics(q);
        
        fid = invDynForm.getFrameId(name);
        oMi = invDynForm.r.framePosition(fid);
        ref_traj = ConstantSE3Trajectory(name, oMi);
        constr = SE3Task(invDynForm.r, fid, ref_traj, name);
        constr.kp = conf.kp_constr;
        constr.kv = conf.kd_constr;
        constr.mask(conf.constraint_mask);

        print "Adding contact", name, ", contact vel", invDynForm.r.frameVelocity(fid).vector[conf.constraint_mask].T;        
        
        Pi = conf.DEFAULT_CONTACT_POINTS;   # contact points is expressed in local frame
        Ni = oMi.rotation.T * conf.DEFAULT_CONTACT_NORMALS; # contact normal is in world frame
        print "    contact point in world frame:", oMi.act_point(Pi).T, (oMi.rotation * Ni).T;
        invDynForm.addUnilateralContactConstraint(constr, Pi, Ni, conf.fMin, conf.mu);
        if(t>0):
            invDynForm.removeTask(name);
    return contact_changed;
    
    
def createSimulator(q0, v0):
    simulator  = Simulator('hrp2_sim'+datetime.now().strftime('%Y%m%d_%H%M%S')+str(np.random.random()), 
                                   q0, v0, conf.fMin, conf.mu, conf.dt, conf.model_path, conf.urdfFileName);
    simulator.viewer.CAMERA_FOLLOW_ROBOT = False;
    simulator.USE_LCP_SOLVER = conf.USE_LCP_SOLVER;
    simulator.ENABLE_TORQUE_LIMITS = conf.FORCE_TORQUE_LIMITS;
    simulator.ENABLE_FORCE_LIMITS = conf.ENABLE_FORCE_LIMITS;
    simulator.ENABLE_JOINT_LIMITS = conf.FORCE_JOINT_LIMITS;
    simulator.ACCOUNT_FOR_ROTOR_INERTIAS = conf.ACCOUNT_FOR_ROTOR_INERTIAS;
    simulator.VIEWER_DT = conf.DT_VIEWER;
    simulator.verb=0;
    return simulator;
        
        
def startSimulation(q0, v0, solverId):
    j = solverId
    print '\nGONNA INTEGRATE CONTROLLER %d' % j;
    
    q[j][:,0] = q0;
    v[j][:,0] = v0;    
    constrViol          = np.empty(conf.MAX_TEST_DURATION).tolist(); #list of lists of lists
    constrViolString    = '';
    torques = np.zeros(na);

    t = 0;
    contact_switch_index = 0;
    simulator.reset(t, q0, v0, conf.dt);
    
    updateConstraints(t, 0, simulator.q, simulator.v, invDynForm, conf.contact_names);
    contact_switch_index += 1;
    contact_names  = [con.name for con in invDynForm.rigidContactConstraints];
    contact_sizes  = [con.dim for con in invDynForm.rigidContactConstraints];
    contact_size_cum = [np.sum(contact_sizes[:ii]) for ii in range(len(contact_sizes))];
    contact_points = [con.framePosition().translation for con in invDynForm.rigidContactConstraints];
    
    for i in range(conf.MAX_TEST_DURATION):            
        invDynForm.setNewSensorData(t, simulator.q, simulator.v);
        (G,glb,gub,lb,ub) = invDynForm.createInequalityConstraints();
        m_in = glb.size;
        
        (D,d)       = invDynForm.computeCostFunction(t);

        q[j][:,i]         = np.matrix.copy(invDynForm.q);
        v[j][:,i]         = np.matrix.copy(invDynForm.v);
        x_com[j][:,i]     = np.matrix.copy(invDynForm.x_com);       # from the solver view-point
        dx_com[j][:,i]    = np.matrix.copy(invDynForm.dx_com);      # from the solver view-point

        if(i%100==0):
            print "Time %.3f... i %d" % (t, i), "Max joint vel %.2f"%np.max(np.abs(v[j][:,i]));
        
        if(i==conf.MAX_TEST_DURATION-1):
            print "MAX TIME REACHED \n";
            print "Max joint vel", np.max(np.abs(v[j][:,i]));
            final_time[j]       = t;
            final_time_step[j]  = i;
            return True;
        
        ''' tell the solvers that if the QP is unfeasible they can relax the joint-acc inequality constraints '''
        solvers[j].setSoftInequalityIndexes(invDynForm.ind_acc_in);
        solvers[j].changeInequalityNumber(m_in);
        (torques, solver_imode[i,j])    = solvers[j].solve(D.A, d.A, G.A, glb.A, gub.A, lb.A, ub.A, torques, maxTime=conf.maxTime);

        tau[j][:,i]                 = np.matrix.copy(torques).reshape((na,1));
        y                           = invDynForm.C * tau[j][:,i] + invDynForm.c;
        dv[j][:,i]                  = y[:nv];
        (tmp1, tmp2, ddx_com[j][:,i]) = invDynForm.r.com(q[j][:,i], v[j][:,i], dv[j][:,i]); #J_com * dv[j][:,i] + invDynForm.dJcom_dq;
        n_active_ineq[i,j]          = solvers[j].nActiveInequalities;   # from the solver view-point
        n_violated_ineq[i,j]        = solvers[j].nViolatedInequalities; # from the solver view-point
#        ineq[i,j,:m_in]             = np.dot(G, tau[i,j,:]) - glb; # from the solver view-point

        if(np.isnan(torques).any() or np.isinf(torques).any()):
            no_sol_count[j] += 1;

        f              = y[nv:nv+invDynForm.k];
        contact_forces = [ f[ii:ii+s] for (ii,s) in zip(contact_size_cum, contact_sizes)];
            
        # impulseDynamics
        constrViol[i] = simulator.integrateAcc(t, dt, dv[j][:,i], fc[j][:,i], tau[j][:,i], conf.PLAY_MOTION_WHILE_COMPUTING);
        
        # display com projection and contact forces in viewer
        simulator.updateComPositionInViewer(np.matrix([x_com[j][0,i], x_com[j][1,i], 0.0]).T);
        simulator.updateContactForcesInViewer(contact_names, contact_points, contact_forces);
        
        for cv in constrViol[i]:
            cv.time = t;
            print cv.toString();
            constrViolString += cv.toString()+'\n';
            
        ''' CHECK TERMINATION CONDITIONS '''
        constraint_errors = [con.positionError(t) for con in invDynForm.rigidContactConstraints];
        for err in constraint_errors:
            if(norm(err) > conf.MAX_CONSTRAINT_ERROR):
                print "ERROR Time %.3f constraint error:"%t, err.T;
                return False;

        ddx_c = invDynForm.Jc * dv[j][:,i] + invDynForm.dJc_v            
        constr_viol = ddx_c - invDynForm.ddx_c_des;
        if(norm(constr_viol)>EPS):
            print "ERROR Time %.3f Constraint violation:"%(t), norm(constr_viol), ddx_c.T, "!=", invDynForm.ddx_c_des.T;
            print "Joint torques:", torques.T
            return False;
            
        # Check whether robot is falling
        if(np.sum(n_violated_ineq[:,j]) > 10 or norm(dx_com[j][:,i])>conf.MAX_COM_VELOCITY):
            print "ERROR Com velocity", np.linalg.norm(dx_com[j][:,i]);
            print "      Solver violated %d inequalities" % solvers[j].nViolatedInequalities; #, "max inequality violation", np.min(ineq[i,j,:m_in]);
            print "      ROBOT FELL AFTER %.3f s\n" % (t);
            final_time[j] = t;
            final_time_step[j] = i;
            for index in range(i+1,conf.MAX_TEST_DURATION):
                q[j][:,index] = q[j][:,i];
            return False;
        t += dt;
        
    return True;


''' *********************** BEGINNING OF MAIN SCRIPT *********************** '''
    
np.set_printoptions(precision=2, suppress=True);
date_time = datetime.now().strftime('%Y%m%d_%H%M%S');
viewer_utils.ENABLE_VIEWER = conf.ENABLE_VIEWER
plot_utils.FIGURE_PATH      = '../results/'+date_time+'/';
plot_utils.SAVE_FIGURES     = conf.SAVE_FIGURES;
plot_utils.SHOW_FIGURES     = conf.SHOW_FIGURES;
plot_utils.SHOW_LEGENDS     = conf.SHOW_LEGENDS;
plot_utils.LINE_ALPHA       = conf.LINE_ALPHA;


T = conf.MAX_TEST_DURATION
print "Max duration of the simulation in time steps", T;

''' CREATE CONTROLLER AND SIMULATOR '''
if(conf.freeFlyer):
    robot = RobotWrapper(conf.urdfFileName, conf.model_path, root_joint=se3.JointModelFreeFlyer());
else:
    robot = RobotWrapper(conf.urdfFileName, conf.model_path, None);
nq = robot.nq;
nv = robot.nv;
dt = conf.dt;
q0 = conf.q0; #se3.randomConfiguration(robot.model, robot.model.lowerPositionLimit, robot.model.upperPositionLimit);
v0 = mat_zeros(nv);
invDynForm = createInvDynFormUtil(q0, v0);
simulator = createSimulator(q0, v0);
robot = invDynForm.r;
na = invDynForm.na;    # number of joints
k = invDynForm.k;    # number of constraints
mass = invDynForm.M[0,0];

''' CREATE POSTURAL TASK '''   
posture_traj = ConstantNdTrajectory("posture_traj", q0[7:,:]);
posture_task = JointPostureTask(invDynForm.r, posture_traj);
posture_task.kp = conf.kp_posture;
posture_task.kv = conf.kd_posture;
invDynForm.addTask(posture_task, conf.w_posture);
    
''' CREATE COM TASK '''
com_ref  = robot.com(q0);
com_ref[0] -= 0.05;
com_traj = ConstantNdTrajectory("com_traj", com_ref);
com_task = CoMTask(invDynForm.r, com_traj);
com_task.kp = conf.kp_com;
com_task.kv = conf.kd_com;
invDynForm.addTask(com_task, conf.w_com);

''' Create the qp solver '''
solver_id       = StandardQpSolver(na, 0, "qpoases", maxIter=conf.maxIter, verb=conf.verb);
solvers = [solver_id];
N_SOLVERS = len(solvers);
solver_names = [s.name for s in solvers];
    
q                    = createListOfMatrices(N_SOLVERS, (nq, conf.MAX_TEST_DURATION));
v                    = createListOfMatrices(N_SOLVERS, (nv, conf.MAX_TEST_DURATION));
fc                   = createListOfMatrices(N_SOLVERS, (k, conf.MAX_TEST_DURATION));
tau                  = createListOfMatrices(N_SOLVERS, (na, conf.MAX_TEST_DURATION));
dv                   = createListOfMatrices(N_SOLVERS, (nv, conf.MAX_TEST_DURATION));
x_com                = createListOfMatrices(N_SOLVERS, (3, conf.MAX_TEST_DURATION));
dx_com               = createListOfMatrices(N_SOLVERS, (3, conf.MAX_TEST_DURATION));
ddx_com              = createListOfMatrices(N_SOLVERS, (3, conf.MAX_TEST_DURATION));
n_active_ineq        = np.zeros((conf.MAX_TEST_DURATION, N_SOLVERS), dtype=np.int);
n_violated_ineq      = np.zeros((conf.MAX_TEST_DURATION, N_SOLVERS), dtype=np.int);
no_sol_count         = np.zeros(N_SOLVERS, dtype=np.int);
solver_imode         = np.zeros((conf.MAX_TEST_DURATION, N_SOLVERS), dtype=np.int);
final_time           = np.zeros(N_SOLVERS);
final_time_step      = np.zeros(N_SOLVERS, np.int);
controller_balance   = N_SOLVERS*[False,];

for s in conf.SOLVER_TO_INTEGRATE:
    controller_balance[s] = startSimulation(q0, v0, s);
#    cProfile.run('startSimulation(q0, v0, s);');

if(conf.PLAY_MOTION_AT_THE_END):
    print "Gonna play computed motion";
    sleep(1);
    simulator.viewer.play(q[s], dt, 1.0);
    print "Computed motion finished";

if(conf.SHOW_FIGURES and conf.PLOT_COM_TRAJ):
    f, ax = plot_utils.create_empty_figure(3, 1);
    ax[0].plot(x_com[s][0,:].A.squeeze(), 'k:');
    ax[1].plot(x_com[s][1,:].A.squeeze(), 'k:');
    ax[2].plot(x_com[s][2,:].A.squeeze(), 'k:');
    ax[0].set_title("Com trajectory");
    plt.show();
