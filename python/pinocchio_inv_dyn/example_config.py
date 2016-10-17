from math import sqrt
import numpy as np
import os

''' *********************** USER-PARAMETERS *********************** '''
SOLVER_ID                   = 0;    # classic TSID formulation
SOLVER_TO_INTEGRATE         = [SOLVER_ID];
DATA_FILE_NAME              = 'data';
TEXT_FILE_NAME              = 'results.txt';
SAVE_DATA                   = False;

''' INITIAL STATE PARAMETERS '''
MAX_TEST_DURATION           = 2000;
dt                          = 1e-3;
model_path                  = [os.getcwd()+"/../../data"];
urdfFileName                = model_path[0] + "/hyq_description/urdf/hyq.urdf";
freeFlyer                   = True;
q0                          = np.matrix([[0.0 ,  0.0,  0.63,  0.  , -0.  ,  0.  ,  1.  , -0.51,  0.74,
                                          -0.93, -0.18, -0.38,  1.35, -0.36,  0.89, -1.19, -0.3 ,  0.19,
                                          0.79]]).T;

''' CONTROLLER CONFIGURATION '''
ENABLE_CAPTURE_POINT_LIMITS     = False;
ENABLE_TORQUE_LIMITS            = True;
ENABLE_FORCE_LIMITS             = True;
ENABLE_JOINT_LIMITS             = True;
IMPOSE_POSITION_BOUNDS          = True;
IMPOSE_VELOCITY_BOUNDS          = True;
IMPOSE_VIABILITY_BOUNDS         = True;
IMPOSE_ACCELERATION_BOUNDS      = True;
JOINT_POS_PREVIEW               = 1.5; # preview window to convert joint pos limits into joint acc limits
JOINT_VEL_PREVIEW               = 1;   # preview window to convert joint vel limits into joint acc limits
MAX_JOINT_ACC                   = 20.0;
MAX_MIN_JOINT_ACC               = 20.0;
USE_JOINT_VELOCITY_ESTIMATOR    = False;
ACCOUNT_FOR_ROTOR_INERTIAS      = True;

# CONTROLLER GAINS
kp_posture  = 30.0; #1.0;   # proportional gain of postural task
kd_posture  = 2*sqrt(kp_posture);
kp_constr   = 100.0;   # constraint proportional feedback gain
kd_constr   = 2*sqrt(kp_constr);   # constraint derivative feedback gain
kp_com      = 30.0;
kd_com      = 2*sqrt(kp_com);
constraint_mask = np.array([True, True, True, False, False, False]).T;

# CONTROLLER WEIGTHS
w_com           = 1e-1;     # weight of the CoM task
w_posture       = 1e-4;     # weight of postural task

# QP SOLVER PARAMETERS
maxIter = 300;      # max number of iterations
maxTime = 0.8;      # max computation time for the solver in seconds
verb=0;             # verbosity level (0, 1, or 2)

# CONTACT PARAMETERS
contact_names = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot'];
DEFAULT_CONTACT_POINTS  = np.matrix([0., 0., 0.]).T    # contact points in local reference frame
DEFAULT_CONTACT_NORMALS = np.matrix([0., 0., 1.]).T    # contact normals in world reference frame
mu  = np.array([0.4, 0.1]);          # force and moment friction coefficient
fMin = 0.0;					     # minimum normal force

# SIMULATOR PARAMETERS
FORCE_TORQUE_LIMITS            = ENABLE_TORQUE_LIMITS;
FORCE_JOINT_LIMITS             = ENABLE_JOINT_LIMITS and IMPOSE_POSITION_BOUNDS;
USE_LCP_SOLVER                 = False

''' STOPPING CRITERIA THRESHOLDS '''
MAX_COM_VELOCITY            = 5;
MAX_CONSTRAINT_ERROR        = 0.1;

''' VIEWER PARAMETERS '''
ENABLE_VIEWER               = True;
PLAY_MOTION_WHILE_COMPUTING = True;
PLAY_MOTION_AT_THE_END      = False;
DT_VIEWER                   = 10*dt;   # timestep used to display motion with viewer

''' FIGURE PARAMETERS '''
SHOW_FIGURES     = True;
PLOT_COM_TRAJ    = True;
SAVE_FIGURES     = False;
SHOW_LEGENDS     = True;
LINE_ALPHA       = 0.7;
