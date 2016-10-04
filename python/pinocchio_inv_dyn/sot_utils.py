# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:23:37 2015

@author: adelpret
"""
#from dynamic_graph.sot.dyninv.meta_task_dyn_6d import MetaTaskDyn6d
import numpy as np
from numpy.linalg import norm
from math import sqrt, atan2, pi
from pinocchio.rpy import rotate
from pinocchio import Quaternion
from pinocchio.utils import matrixToRpy, rpyToMatrix
from polytope_conversion_utils import *
from qpoases import PyQProblemB as QProblemB # QP with simple bounds only
from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue

EPS = 1e-6

RIGHT_FOOT_SIZES  = (0.130,  0.100,  0.056,  0.075); # pos x, neg x, pos y, neg y size 
LEFT_FOOT_SIZES = (0.130,  0.100,  0.075,  0.056); # pos x, neg x, pos y, neg y size 
H_FOOT_2_SOLE = ((1, 0, 0, 0),(0, 1, 0, 0), (0, 0, 1, -0.105),(0, 0, 0, 1));
H_RFOOT_2_FR_CORNER = ((1, 0, 0, RIGHT_FOOT_SIZES[0]),  (0, 1, 0,-RIGHT_FOOT_SIZES[3]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # front right
H_RFOOT_2_FL_CORNER = ((1, 0, 0, RIGHT_FOOT_SIZES[0]),  (0, 1, 0, RIGHT_FOOT_SIZES[2]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # front left
H_RFOOT_2_HR_CORNER = ((1, 0, 0, -RIGHT_FOOT_SIZES[1]),(0, 1, 0,-RIGHT_FOOT_SIZES[3]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # hind right
H_RFOOT_2_HL_CORNER = ((1, 0, 0, -RIGHT_FOOT_SIZES[1]),(0, 1, 0, RIGHT_FOOT_SIZES[2]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # hind left
H_LFOOT_2_FR_CORNER = ((1, 0, 0, LEFT_FOOT_SIZES[0]),  (0, 1, 0,-LEFT_FOOT_SIZES[3]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # front right
H_LFOOT_2_FL_CORNER = ((1, 0, 0, LEFT_FOOT_SIZES[0]),  (0, 1, 0, LEFT_FOOT_SIZES[2]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # front left
H_LFOOT_2_HR_CORNER = ((1, 0, 0, -LEFT_FOOT_SIZES[1]),(0, 1, 0,-LEFT_FOOT_SIZES[3]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # hind right
H_LFOOT_2_HL_CORNER = ((1, 0, 0, -LEFT_FOOT_SIZES[1]),(0, 1, 0, LEFT_FOOT_SIZES[2]), (0, 0, 1, -0.105),(0, 0, 0, 1));   # hind left
H_WRIST_2_GRIPPER = ((1, 0, 0, 0.09),(0, 1, 0, 0.02), (0, 0, 1, -0.340),(0, 0, 0, 1));
#H_WRIST_2_GRIPPER = ((1, 0, 0, 0),(0, 1, 0, 0), (0, 0, 1, -0.18),(0, 0, 0, 1));

Q_MAX = array([0.5236, 0.3491,  0.733 ,  2.618 ,  0.733 ,  0.6109,  
               0.7854,  0.6109, 0.733 ,  2.618 ,  0.733 ,  0.3491,  
               0.7854,  1.0472,  0.7854, 0.7854,  
               1.0472,  0.1745,  1.6057,  0.0349,  1.6057,  1.6057, 0.7854,  
               1.0472,  1.6581,  1.6057,  0.0349,  1.6057,  1.6057, 0.7854])
               
Q_MIN = np.array([-0.7854, -0.6109, -2.1817, -0.0349, -1.309 , -0.3491, 
                  -0.5236, -0.3491, -2.1817, -0.0349, -1.309 , -0.6109, 
                  -0.7854, -0.0873, -0.7854, -0.5236, 
                  -3.1416, -1.6581, -1.6057, -2.3911, -1.6057, -1.6057, 0., 
                  -3.1416, -0.1745, -1.6057, -2.3911, -1.6057, -1.6057, 0. ])
        
Q_HALF_SITTING = (  # Free flyer
                    0., 0., 0.648702, 0., 0. , 0.,
                    # Legs
                    0., 0., -0.453786, 0.872665, -0.418879, 0.,
                    0., 0., -0.453786, 0.872665, -0.418879, 0.,
                    # Chest and head
                    0., 0., 0., 0.,
                    # Arms
                    0.261799, -0.17453, 0., -0.523599, 0., 0., 0.1,
                    0.261799, 0.17453,  0., -0.523599, 0., 0., 0.1);
                    
DQ_MAX = (3.54108, 3.80952, 4.42233, 3.9801, 5.07937, 9.14286, 
          3.54108, 3.80952, 4.42233, 3.9801, 5.07937, 9.14286, 
          4.40217, 2.3963, 6.87679, 6.87679,
          4.17044, 2.4192, 4.15699, 2.23776, 4.72794, 3.16727, 5.68126,
          4.17044, 2.4192, 4.15699, 2.23776, 4.72794, 3.16727, 5.68126);

GEAR_RATIO = (0,0,0,0,0,0,
              384.0, 240.0, 180.0, 200.0, 180.0, 100.0,
              384.0, 240.0, 180.0, 200.0, 180.0, 100.0,
              207.69, 381.54, 100.0, 100.0,
              219.23, 231.25, 266.67, 250.0, 145.45, 350.0, 200.0,
              219.23, 231.25, 266.67, 250.0, 145.45, 350.0, 200.0);
INERTIA_ROTOR = (0,0,0,0,0,0,
                 1.01e-6, 6.96e-6, 1.34e-6, 1.34e-6, 6.96e-6, 6.96e-6,
                 1.01e-6, 6.96e-6, 1.34e-6, 1.34e-6, 6.96e-6, 6.96e-6,
                 6.96e-6, 6.96e-6, 1.10e-6, 1.10e-6,
                 6.96e-6, 6.60e-6, 1.00e-6, 6.60e-6, 1.10e-6, 1.00e-6, 1.00e-6, 
                 6.96e-6, 6.60e-6, 1.00e-6, 6.60e-6, 1.10e-6, 1.00e-6, 1.00e-6);
#INERTIA_ROTOR = (0,0,0,0,0,0,1.01e-4,6.96e-4,1.34e-4,1.34e-4,6.96e-4,6.96e-4,1.01e-4,6.96e-4,1.34e-4,1.34e-4,6.96e-4,6.96e-4,6.96e-4,6.96e-4,1.10e-4,1.10e-4,6.96e-4,6.60e-4,1.00e-4,6.60e-4,1.10e-4,1.00e-4,1.00e-4,1.00e-4,6.96e-4,6.60e-4,1.00e-4,6.60e-4,1.10e-4,1.00e-4,1.00e-4,1.00e-4);

TAU_MAX = np.array([86, 175, 151, 168, 131, 73, 
                    86, 175, 151, 168, 131, 73, 
                    151, 278, 9.6, 9.6, 
                    160, 100, 48, 108, 13.9, 63, 3,
                    160, 100, 48, 108, 13.9, 63, 3]);

JOINT_VISCOUS_FRICTION = 6*(0,) + 30*(0,);

#
#''' Convert from Roll, Pitch, Yaw to transformation Matrix. '''
#def rpyToMatrix(rpy):
#    return rotate('z',rpy[2])*rotate('y',rpy[1])*rotate('x',rpy[0]);
#
#''' Convert from Transformation Matrix to Roll, Pitch, Yaw '''
#def matrixToRpy(M):
#    m = sqrt(M[2, 1] ** 2 + M[2, 2] ** 2)
#    p = atan2(-M[2, 0], m)
#
#    if abs(abs(p) - pi / 2) < 0.001:
#        r = 0
#        y = -atan2(M[0, 1], M[1, 1])
#    else:
#        y = atan2(M[1, 0], M[0, 0])  # alpha
#        r = atan2(M[2, 1], M[2, 2])  # gamma
#
#    return np.array([r, p, y]);
    
    
def pinocchio_2_sot(q):
    # PINOCCHIO Free flyer 0-6, CHEST HEAD 7-10, LARM 11-17, RARM 18-24, LLEG 25-30, RLEG 31-36
    # SOT       Free flyer 0-5, RLEG 6-11, LLEG 12-17, CHEST HEAD 18-21, RARM 22-28, LARM 29-35
    qSot = np.matlib.zeros((36,1));
    qSot[:3] = q[:3];
    quatMat = Quaternion(q[6,0], q[3,0], q[4,0], q[5,0]).matrix();
    qSot[3:6] = matrixToRpy(quatMat);
    qSot[18:22] = q[7:11]; # chest-head
    qSot[29:]   = q[11:18]; # larm
    qSot[22:29] = q[18:25]; # rarm
    qSot[12:18] = q[25:31]; # lleg
    qSot[6:12]  = q[31:]; # rleg
    return qSot.A.squeeze();
    
def sot_2_pinocchio(q):
    # PINOCCHIO Free flyer 0-6, CHEST HEAD 7-10, LARM 11-17, RARM 18-24, LLEG 25-30, RLEG 31-36
    # SOT       Free flyer 0-5, RLEG 6-11, LLEG 12-17, CHEST HEAD 18-21, RARM 22-28, LARM 29-35
    qPino = np.matlib.zeros((37,1));
    qPino[:3,0] = q[:3];
    quatMat = rpyToMatrix(q[3:6]);
    quatVec = Quaternion(quatMat);
    qPino[3:7,0]   = quatVec.coeffs();
    qPino[7:11,0]  = q[18:22]; # chest-head
    qPino[11:18,0] = q[29:]; # larm
    qPino[18:25,0] = q[22:29]; # rarm
    qPino[25:31,0] = q[12:18]; # lleg
    qPino[31:,0]   = q[6:12]; # rleg
    return qPino;
    
def sot_2_pinocchio_vel(v):
    vPino = np.zeros(36);
    vPino[:6] = v[:6];
    vPino[6:10]  = v[18:22]; # chest-head
    vPino[10:17] = v[29:]; # larm
    vPino[17:24] = v[22:29]; # rarm
    vPino[24:30] = v[12:18]; # lleg
    vPino[30:]   = v[6:12]; # rleg
    return vPino;

# CROSSMATRIX Compute the projection matrix of the cross product
def crossMatrix( v ):
    VP = np.array( [[  0,  -v[2], v[1] ],
                    [ v[2],  0,  -v[0] ],
                    [-v[1], v[0],  0   ]] );
    return VP;
    
''' Compute the inequality constraints of the 6D wrench applicable to 
    an arbitrary set of contact points with a friction coefficient of mu.
        H w <= 0 
'''
def computeContactInequalities(contact_points, contact_normals, mu):
    c = contact_points.shape[0];    # number of contact points
    cg = 4;                         # number of generators per contact point    
    G4 = np.zeros((c,3,cg));        # contact generators
    G_centr4 = np.zeros((6,c*cg));
    ''' contact positions '''
    p = np.asarray(contact_points); 
    ''' contact normal and tangential directions '''
    N = np.asarray(contact_normals);
    T1 = np.empty((c,3));
    T2 = np.empty((c,3));
    muu = mu/sqrt(2);
    for i in range(c):
        T1[i,:] = np.cross(N[i,:], np.array([1,0,0]));
        if(norm(T1[i,:]) < EPS):
            T1[i,:] = np.cross(N[i,:], np.array([0,1,0]));
        T1[i,:] /= norm(T1[i,:]);
        T2[i,:] = np.cross(N[i,:], T1[i,:]);
        T2[i, :] /= norm(T2[i, :]);
        
        ''' compute generators '''            
        G4[i,:,0] =  muu*T1[i,:] + muu*T2[i,:] + N[i,:];
        G4[i,:,1] =  muu*T1[i,:] - muu*T2[i,:] + N[i,:];
        G4[i,:,2] = -muu*T1[i,:] + muu*T2[i,:] + N[i,:];
        G4[i,:,3] = -muu*T1[i,:] - muu*T2[i,:] + N[i,:];

        ''' project generators in 6d centroidal space '''
        G_centr4[:3,cg*i:cg*i+cg] = G4[i,:,:];
        G_centr4[3:,cg*i:cg*i+cg] = np.dot(crossMatrix(p[i,:]), G4[i,:,:]);
        
    ''' convert generators to inequalities '''
    H = cone_span_to_face(G_centr4);
    return H;
    
''' Compute the inequality constraints of the 6D wrench applicable to a rectangular
    surface of dimension (lxp+lxn)x(lyp+lyn) with a friction coefficient of mu.
        H w <= 0 
'''
def computeRectangularContactInequalities(lxp, lxn, lyp, lyn, mu):
    c = 4;              # number of contact points
    cg = 4;             # number of generators per contact point    
    G4 = np.zeros((3,cg));
    G_centr4 = np.zeros((6,c*cg));
    ''' foot corner positions '''
    p = np.array([[ lxp,  lyp, 0],
               [ lxp, -lyn, 0],
               [-lxn, -lyn, 0],
               [-lxn,  lyp, 0]]);
    ''' compute generators '''        
    muu = mu/sqrt(2);
    G4[:,0] = np.array([ muu, muu, 1]);
    G4[:,1] = np.array([ muu,-muu, 1]);
    G4[:,2] = np.array([-muu, muu, 1]);
    G4[:,3] = np.array([-muu,-muu, 1]);
#    G4[:,0] = np.array([ mu, 0, 1]);
#    G4[:,1] = np.array([-mu, 0, 1]);
#    G4[:,2] = np.array([ 0, mu, 1]);
#    G4[:,3] = np.array([ 0,-mu, 1]);
    ''' project generators in 6d centroidal space '''
    for i in range(c):
        G_centr4[:3,cg*i:cg*i+cg] = G4;
        G_centr4[3:,cg*i:cg*i+cg] = np.dot(crossMatrix(p[i,:]), G4);
    ''' convert generators to inequalities '''
    H = cone_span_to_face(G_centr4);
    return H;
    
''' Solve the underconstrained system of linear equalities A*x=b adding the 
    specified vector x0 to the null space of A:
        x = A^+ * b + N_A * x0
    where N_A is an orthogonal projector in the null space of A '''
def solveWithNullSpace(A, b, x0, rtol=1e-5):
    U, s, VT = np.linalg.svd(A);
    rank = (s > rtol).sum();
    x = np.dot(VT[:rank].T, (1/s)*np.dot(U.T,b));
    x += np.dot(VT[rank:].T, np.dot(VT[rank:], x0));
    return x;
    
def compute_nullspace_projector(A, rtol=1e-5):
    U, s, VT = np.linalg.svd(A);
    rank = (s > rtol).sum();
    return np.dot(VT[rank:].T, VT[rank:]);

def compute_pinv_and_nullspace_projector(A, rtol=1e-5):
    U, s, VT = np.linalg.svd(A);
    rank = (s > rtol).sum();
    Ap = np.dot(VT[:rank].T, (1/s)*U.T);
    N = np.dot(VT[rank:].T, VT[rank:])
    return (Ap, N);

''' Solve a hierarchy of least-squares problem of the form
        minimize    || A_i x - b_i ||^2
        subject to   A_j x = A_j x_j       for all j<i
    where x_j is the optimum computed optimizing level j.
'''
def solveHierarchicalLeastSquares(A_list, b_list, damping=1e-4, zero_thr=1e-5):
    n = A_list[0].shape[1];
    N = np.identity(n);
    x = np.zeros(n);
    for i in range(len(A_list)):
        if(A_list[i].shape[0] != b_list[i].shape[0]):
            print "[solveHierarchicalLeastSquares] ERROR shape of A[%d] and b[%d] do not match"%(i,i), A_list[i].shape[0], b_list.shape[0];
            return None;
        if(A_list[i].shape[1] != n):
            print "[solveHierarchicalLeastSquares] ERROR shape of A[%d] do not match shape of A[0]"%(i,i), A_list[i].shape[1], n;
            return None
        A = np.dot(A_list[i], N);
        b = b_list[i] - np.dot(A_list[i], x);
        U, s, VT = np.linalg.svd(A, full_matrices=False);
        x += np.dot(VT.T, (s/(s**2+damping**2))*np.dot(U.T,b));
        rank = (s > zero_thr).sum();
        N = np.dot(N, np.dot(VT[rank:].T, VT[rank:]));
    return x;

def qpOasesSolverMsg(imode):
    if(imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY):
        return "HOTSTART_STOPPED_INFEASIBILITY";
    elif(imode==PyReturnValue.MAX_NWSR_REACHED):
        return "RET_MAX_NWSR_REACHED";
    elif(imode==PyReturnValue.STEPDIRECTION_FAILED_TQ):
        return "STEPDIRECTION_FAILED_TQ";
    elif(imode==PyReturnValue.STEPDIRECTION_FAILED_CHOLESKY):
        return "STEPDIRECTION_FAILED_CHOLESKY";
    elif(imode==PyReturnValue.HOTSTART_FAILED_AS_QP_NOT_INITIALISED):
        return "HOTSTART_FAILED_AS_QP_NOT_INITIALISED"; # 54
    elif(imode==PyReturnValue.INIT_FAILED_HOTSTART):
        return "INIT_FAILED_HOTSTART"; # 36
    elif(imode==PyReturnValue.INIT_FAILED_INFEASIBILITY):
        return "INIT_FAILED_INFEASIBILITY"; # 37
    elif(imode==PyReturnValue.UNKNOWN_BUG):
        return "UNKNOWN_BUG"; # 9
    return str(imode);

''' Solve the least square problem:
    minimize    || A*x-b ||^2
    subject to  lb_in <= A_in*x <= ub_in
                   lb <= x <= ub
'''
def solveLeastSquare(A, b, lb=None, ub=None, A_in=None, lb_in=None, ub_in=None, maxIterations=None, maxComputationTime=60.0, regularization=1e-8):
    n = A.shape[1];
    m_in = 0;
    if(A_in!=None):
        m_in = A_in.shape[0];
        if(lb_in==None):
            lb_in = np.array(m_in*[-1e99]);
        if(ub_in==None):
            ub_in = np.array(m_in*[1e99]);
    if(lb==None):
        lb = np.array(n*[-1e99]);
    if(ub==None):
        ub = np.array(n*[1e99]);

    # 0.5||Ax-b||^2 = 0.5(x'A'Ax - 2b'Ax + b'b) = 0.5x'A'Ax - b'Ax +0.5b'b
    Hess = np.dot(A.T,A) + regularization*np.identity(n);
    grad = -np.dot(A.T,b);
    if(maxIterations==None):
        maxActiveSetIter = np.array([100+2*m_in+2*n]);
    else:
        maxActiveSetIter = np.array([maxIterations]);
    maxComputationTime = np.array([maxComputationTime]);
    options = Options();
    options.printLevel = PrintLevel.NONE; #NONE, LOW, MEDIUM
    options.enableRegularisation = False;
    if(m_in==0):
        qpOasesSolver = QProblemB(n); #, HessianType.SEMIDEF);
        qpOasesSolver.setOptions(options);
        # beware that the Hessian matrix may be modified by this function
        imode = qpOasesSolver.init(Hess, grad, lb, ub, maxActiveSetIter, maxComputationTime);
    else:
        qpOasesSolver = SQProblem(n, m_in); #, HessianType.SEMIDEF);
        qpOasesSolver.setOptions(options);
        imode = qpOasesSolver.init(Hess, grad, A_in, lb, ub, lb_in, ub_in, maxActiveSetIter, maxComputationTime);
    x = np.empty(n);
    qpOasesSolver.getPrimalSolution(x);
    #print "QP cost:", 0.5*(np.linalg.norm(np.dot(A, x)-b)**2);
    return (imode,x);
    
    
def setDynamicProperties(model, dt):
    model.setProperty('TimeStep', str(dt))

    model.setProperty('ComputeAcceleration', 'false')
    model.setProperty('ComputeAccelerationCoM', 'false')
    model.setProperty('ComputeBackwardDynamics', 'false')
    model.setProperty('ComputeCoM', 'false')
    model.setProperty('ComputeMomentum', 'false')
    model.setProperty('ComputeSkewCom', 'false')
    model.setProperty('ComputeVelocity', 'false')
    model.setProperty('ComputeZMP', 'false')

    model.setProperty('ComputeAccelerationCoM', 'true')
    model.setProperty('ComputeCoM', 'true')
    model.setProperty('ComputeVelocity', 'true')
    model.setProperty('ComputeZMP', 'true')

''' Create and initialize a MetaTaskDyn6d.
    - taskName is a unique string ideintifying the task
    - dynamic is an instance of the sot Dynamic entity
    - opPointName is the name of the body controlled by the task
    - opPointMod is a homogenous matrix in tuple form which maps the body reference to the controlled point
    - ctrlGain is the gain associated to the task
'''
def createAndInitializeMetaTaskDyn6D(taskName, dt, dynamic, opPointName, opPointMod=None, ctrlGain=None, selecFlag='111111'):
    task = MetaTaskDyn6d(taskName, dynamic, opPointName, opPointName);
    task.task.dt.value = dt;
    if(opPointMod==None):
        task.opmodif = ((1, 0, 0, 0),(0, 1, 0, 0), (0, 0, 1, 0),(0, 0, 0, 1));
    else:
        task.opmodif = opPointMod;
    task.opPointModif.position.recompute(0);
    task.ref = task.opPointModif.position.value;
    if(ctrlGain!=None):
        task.task.controlGain.value = ctrlGain;
    task.feature.errordot.value = (0,0,0,0,0,0);
    task.feature.selec.value = selecFlag;
    task.feature.frame("desired");
    task.task.task.recompute(0);
    task.task.jacobian.recompute(0);
    return task;
    
def hrp2_jointId_2_name(jid):
    switcher = {
        0: "rhy",
        1: "rhr",
        2: "rhp",
        3: "rk",
        4: "rap",
        5: "rar",
        6: "lhy",
        7: "lhr",
        8: "lhp",
        9: "lk",
        10: "lap",
        11: "lar",
        12: "ty",
        13: "tp",
        14: "hy",
        15: "hp",
        16: "rsp",
        17: "rsr",
        18: "rsy",
        19: "re",
        20: "rwy",
        21: "rwp",
        22: "rh",
        23: "lsp",
        24: "lsr",
        25: "lsy",
        26: "le",
        27: "lwy",
        28: "lwp",
        29: "lh"        
    };
    return switcher.get(jid, "nothing");
    
if __name__=="__main__":
    q =   np.array( [   0.0, 0.0, 0.648702, 0.0, 0.0 , 0.0, 1.0,                             # Free flyer 0-6
                        0.0, 0.0, 0.0, 0.0,                                                  # CHEST HEAD 7-10
                        0.261799388,  0.174532925, 0.0, -0.523598776, 0.0, 0.0, 0.174532925, # LARM       11-17
                        0.261799388, -0.174532925, 0.0, -0.523598776, 0.0, 0.0, 0.174532925, # RARM       18-24
                        0.0, 0.0, -0.453785606, 0.872664626, -0.41887902, 0.0,               # LLEG       25-30
                        0.0, 0.0, -0.453785606, 0.872664626, -0.41887902, 0.0,               # RLEG       31-36
                        ]);
    qSot = pinocchio_2_sot(q);
    qPino = sot_2_pinocchio(qSot);
    if(np.linalg.norm(q-qPino) > 1e-6):
        print "Error in conversion pino-sot-pino"
        print q
        print qPino
        
    q =  np.array([ 0.  ,  0.  ,  0.62,  0.5  ,  0.3  ,  -0.2  ,  0.  ,  0.  , -0.15,
                        0.87, -0.72,  0.  ,  0.  ,  0.  , -0.75,  0.87, -0.12,  0.  ,
                        0.  ,  0.  ,  0.  ,  0.  ,  0.26, -0.17,  0.  , -0.52,  0.  ,
                        0.  ,  0.1 ,  0.26,  0.17,  0.  , -0.52,  0.  ,  0.  ,  0.1 ]);
    qPino = sot_2_pinocchio(q);
    qSot  = pinocchio_2_sot(qPino);
    if(np.linalg.norm(q-qSot) > 1e-6):
        print "Error in conversion sot-pino-sot"
        print q
        print qSot