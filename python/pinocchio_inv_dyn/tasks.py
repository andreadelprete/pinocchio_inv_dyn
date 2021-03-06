#!/usr/bin/env python

#from tools import *
import pinocchio as se3
from pinocchio.utils import zero as mat_zeros
from pinocchio.utils import cross as mat_cross
from pinocchio import SE3
from pinocchio import Motion
import numpy as np
from numpy.linalg import norm
import time
from math import *

EPS = 1e-5;

def errorInSE3 (M, Mdes):
	error = se3.log(Mdes.inverse () * M)
	return error

# Define a generic Task
class Task:
  def __init__ (self, robot, name = "Task"):
    self.name = name
    self.robot = robot
    self.kp = 1.
    self.kv = 1.
    self._coeff = 1.

  def setCoeff(self, value):
    self._coeff = value

  def dyn_value(self, t, q, v, update_geometry = False):
    a_des = np.matrix ([]).reshape (0, 0)
    J = np.matrix ([]).reshape (0, 0)
    drift = np.matrix ([]).reshape (0, 0)
    return J, drift, a_des

  def jacobian (self):
    jacobian = matrix ([]).reshape (0, robot.nq)
    return jacobian
  
  @property
  def dim(self):
    return 0

# Define SE3 Task
class SE3Task(Task):

  def __init__ (self, robot, frame_id, ref_trajectory, name = "SE3 Task"):
    Task.__init__ (self, robot, name)
    self._frame_id = frame_id
    self._ref_trajectory = ref_trajectory

    # set default value to M_ref
    self._M_ref = SE3.Identity () 
  
    # mask over the desired euclidian axis
    self._mask = (np.ones(6)).astype(bool)
    self._gMl = SE3.Identity();

  @property
  def dim(self):
    return self._mask.sum ()

  def mask(self, mask):
    assert len(mask) == 6, "The mask must have 6 elemets"
    self._mask = mask.astype(bool)

  @property
  def refTrajectory(self):
    return self._ref_trajectory

  def refConfiguration (self, M_ref):
    assert isinstance(M_ref, SE3), "M_ref is not an element of class SE3"
    self._M_ref = M_ref
    
  def framePosition(self):
    return self.robot.framePosition(self._frame_id);
    
  def positionError(self, t):
    oMi = self.robot.framePosition(self._frame_id);
    M_ref, v_ref, a_ref  = self._ref_trajectory(t);
    p_error = errorInSE3(oMi, M_ref);
    return p_error.vector[self._mask];
    
  def velocityError(self, t):
    oMi = self.robot.framePosition(self._frame_id);
    self._gMl.rotation = oMi.rotation
    v_frame = self.robot.frameVelocity(self._frame_id);
    M_ref, v_ref, a_ref  = self._ref_trajectory(t);
    v_error = v_frame - self._gMl.actInv(v_ref)
    return v_error.vector[self._mask];

  def dyn_value(self, t, q, v, local_frame = True):
    # Get the current configuration of the link
    oMi = self.robot.framePosition(self._frame_id);
    v_frame = self.robot.frameVelocity(self._frame_id)
    
    # Get the reference trajectory
    M_ref, v_ref, a_ref  = self._ref_trajectory(t)

    # Transformation from local to world    
    self._gMl.rotation = oMi.rotation 

    # Compute error acceleration desired
    p_error= errorInSE3(oMi, M_ref);
    v_error = v_frame - self._gMl.actInv(v_ref)
    drift = self.robot.frameAcceleration(self._frame_id)
    drift.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T    
    a_des = -self.kp * p_error.vector -self.kv * v_error.vector + self._gMl.actInv(a_ref).vector

    J = self.robot.frameJacobian(q, self._frame_id, False)
    
    if(local_frame==False):
        drift = self._gMl.act(drift);
        a_des[:3] = self._gMl.rotation * a_des[:3];
        a_des[3:] = self._gMl.rotation * a_des[3:];
        J[:3,:] = self._gMl.rotation * J[:3,:];
        J[3:,:] = self._gMl.rotation * J[3:,:];
        
    return J[self._mask,:], drift.vector[self._mask], a_des[self._mask]

  def jacobian(self, q, update_geometry = False):
    self.__jacobian_value = self.robot.frameJacobian(q, self._frame_id, update_geometry)
    
    return self.__jacobian_value[self._mask,:] 


# Define CoM Task
class CoMTask(Task):

  def __init__ (self, robot, ref_trajectory, name = "CoM Task"):
    assert ref_trajectory.dim == 3
    Task.__init__ (self, robot, name)
    self._ref_trajectory = ref_trajectory

    # mask over the desired euclidian axis
    self._mask = (np.ones(3)).astype(bool)

  #@kp.setter
  #def kp(self, *args):
  #  num_args = len(args)
  #  self._kp = args
  #  #if num_args == 1:
  #  #  if isinstance(args, np.matrix):
  #  #    self._kp = np.diag(args)
  #  #  else:
  #  #    self._kp = args
  #  #else:
  #  #  assert num_args == 3
  #  #  self._kp = np.diag(args)

  @property
  def dim(self):
    return self._mask.sum()

  @property
  def refTrajectory(self):
    return self._ref_trajectory

  def mask(self, mask):
    assert len(mask) == 3, "The mask must have 3 elements"
    self._mask = mask.astype(bool)
    
  def dyn_value(self, t, q, v, update_geometry = False):
    # Get the current CoM position, velocity and acceleration
    p_com, v_com, a_com = self.robot.com(q,v,0*v)

    # Get reference CoM trajectory
    p_ref, v_ref, a_ref = self._ref_trajectory(t)

    # Compute errors
    p_error = p_com - p_ref
    v_error = v_com - v_ref 

    drift = a_com # Coriolis acceleration
    a_des = -(self.kp * p_error + self.kv * v_error) + a_ref

    # Compute jacobian
    J = self.robot.Jcom(q)

    return J[self._mask,:], drift[self._mask], a_des[self._mask]

  def jacobian(self, q, update_geometry = True):    
    self.__jacobian_value = self.robot.Jcom(q) # TODO - add update geometry option
    return self.__jacobian_value[self._mask,:] 


''' CoM Task with acceleration scaling. 
    This means that if the CoM desired acceleration is not feasible,
    the solver will try to get as close as possible to the desired value,
    while not altering too much the CoM acceleration direction.
'''
class CoMTaskWithAccelerationScaling(Task):

  def __init__ (self, robot, ref_trajectory, weight, name = "CoM Task"):
    assert ref_trajectory.dim == 3
    Task.__init__ (self, robot, name)
    self._ref_trajectory = ref_trajectory
    # mask over the desired euclidian axis
    self._mask = (np.ones(3)).astype(bool)
    self.weight = weight

  @property
  def dim(self):
    return self._mask.sum()

  @property
  def refTrajectory(self):
    return self._ref_trajectory

  def mask(self, mask):
    assert len(mask) == 3, "The mask must have 3 elements"
    self._mask = mask.astype(bool)
    
  def dyn_value(self, t, q, v, update_geometry = False):
    # Get the current CoM position, velocity and acceleration
    p_com, v_com, a_com = self.robot.com(q,v,0*v)

    # Get reference CoM trajectory
    p_ref, v_ref, a_ref = self._ref_trajectory(t)

    # Compute errors
    p_error = p_com - p_ref
    v_error = v_com - v_ref 

    drift = a_com # Coriolis acceleration
    a_des = -(self.kp * p_error + self.kv * v_error) + a_ref

    # Compute jacobian
    J = self.robot.Jcom(q)

    D = mat_zeros((3,3));
    D[:,0] = a_des / norm(a_des);
    # compute 2 directions orthogonal to D[0,:] and orthogonal to each other
    D[:,1] = mat_cross(D[:,0], np.matrix([0.0, 0.0, 1.0]).T);
    if(norm(D[:,1])<EPS):
      D[:,1] = mat_cross(D[:,0], np.matrix([0.0, 1.0, 0.0]).T);
    D[:,1] *= self.weight/norm(D[:,1]);
    D[:,2] = mat_cross(D[:,0], D[:,1]);
    D[:,2] *= self.weight/norm(D[:,2]);

    J     = D.T*J;
    drift = D.T*drift;
    a_des = D.T*a_des;

    return J[self._mask,:], drift[self._mask], a_des[self._mask]


''' Task to maintain a straight CoM path.
'''
class CoMStraightPathTask(Task):

  def __init__ (self, robot, name = "CoM Straight Path Task"):
    Task.__init__ (self, robot, name)
    # mask over the desired euclidian axis
    self._mask = (np.ones(2)).astype(bool)

  @property
  def dim(self):
    return self._mask.sum()

  def mask(self, mask):
    assert len(mask) == 2, "The mask must have 2 elements"
    self._mask = mask.astype(bool)
    
  def dyn_value(self, t, q, v, update_geometry = False):
    # Get the current CoM position, velocity and acceleration
    p_com, v_com, drift = self.robot.com(q,v,0*v)
    a_des = mat_zeros(2);
    # Compute jacobian
    J = self.robot.Jcom(q)

    # compute 2 directions orthogonal to v_com and orthogonal to each other
    D = mat_zeros((3,2));
    d = v_com / norm(v_com);
    D[:,0] = mat_cross(d, np.matrix([0.0, 0.0, 1.0]).T);
    if(norm(D[:,0])<EPS):
      D[:,0] = mat_cross(d, np.matrix([0.0, 1.0, 0.0]).T);
    D[:,0] /= norm(D[:,0]);
    D[:,1] = mat_cross(d, D[:,0]);
    D[:,1] /= norm(D[:,1]);

    J     = D.T*J;
    drift = D.T*drift;

    return J[self._mask,:], drift[self._mask], a_des[self._mask]
    

''' Define Postural Task considering only the joints (and not the floating base). '''
class JointPostureTask(Task):

  def __init__ (self, robot, ref_trajectory, name = "Joint Posture Task"):
    Task.__init__ (self, robot, name)

    # mask over the desired euclidian axis
    self._mask = (np.ones(robot.nv-6)).astype(bool)

    # desired postural configuration
    self._ref_traj = ref_trajectory;

    # Init
    self._jacobian = np.matlib.zeros((robot.nv-6, robot.nv));
    self._jacobian[:,6:] = np.matlib.identity((robot.nv-6));

  @property
  def dim(self):
    return self._mask.sum ()

  def mask(self, mask):
    assert len(mask) == self.robot.nv-6, "The mask must have {} elements".format(self.robot.nv-6)
    self._mask = mask.astype(bool)

  def dyn_value(self, t, q, v, update_geometry = False):
    # Compute error
    (q_ref, v_ref, a_ref) = self._ref_traj(t);
    err = q[7:,0] - q_ref;
    derr = v[6:, 0] - v_ref;
    self.a_des = a_ref -(self.kp * err + self.kv * derr);
    self.drift = 0*self.a_des;
    
    return self._jacobian[self._mask,:], self.drift[self._mask], self.a_des[self._mask]


''' Define Postural Task '''
class PosturalTask(Task):

  def __init__ (self, robot, name = "Postural Task"):
    Task.__init__ (self, robot, name)

    # mask over the desired euclidian axis
    self._mask = (np.ones(robot.nv)).astype(bool)

    # desired postural configuration
    self.q_posture_des = zero(robot.nq)

    # Init
    self.__error_value = np.matrix(np.empty([robot.nv,1]))
    self.__jacobian_value = np.matrix(np.identity(robot.nv))
    self.__gain_vector = np.matrix(np.ones([1.,robot.nv]))

  @property
  def dim(self):
    return self._mask.sum ()

  def setPosture(self, q_posture):
    self.q_posture_des = np.matrix.copy(q_posture);

  def setGain(self, gain_vector):
    assert gain_vector.shape == (1, self.robot.nv) 
    self.__gain_vector = np.matrix(gain_vector)

  def getGain(self):
    return self.__gain_vector

  def mask(self, mask):
    assert len(mask) == self.robot.nv, "The mask must have {} elements".format(self.robot.nq)
    self._mask = mask.astype(bool)

  def error_dyn(self, t, q, v):
    M_ff = XYZQUATToSe3(q[:7])
    M_ff_des = XYZQUATToSe3(self.q_posture_des[:7])
    error_ff = errorInSE3(M_ff, M_ff_des).vector() 
    
    # Compute error
    error_value = self.__error_value
    error_value[:6,0] = error_ff
    error_value[6:,0] = q[7:,0] - self.q_posture_des[7:,0]
 
    return error_value[self._mask], v[self._mask], 0.

  def dyn_value(self, t, q, v, update_geometry = False):
    M_ff = XYZQUATToSe3(q[:7])
    M_ff_des = XYZQUATToSe3(self.q_posture_des[:7])
    error_ff = errorInSE3(M_ff, M_ff_des).vector
    
    # Compute error
    error_value = self.__error_value
    error_value[:6,0] = error_ff
    error_value[6:,0] = q[7:,0] - self.q_posture_des[7:,0]
    
    self.J = np.diag(self.__gain_vector.A.squeeze())
    self.a_des = -(self.kp * error_value + self.kv * v)
    self.drift = 0*self.a_des
    
    return self.J[self._mask,:], self.drift[self._mask], self.a_des[self._mask]

  def jacobian(self, q):
    self.__jacobian_value = np.diag(self.__gain_vector.A.squeeze())
    return self.__jacobian_value[self._mask,:] 

# Define Angular Momentum Task
class AngularMomentumTask(Task):

  def __init__ (self, robot, refTraj, name = "Angular Momentum Task"):
    Task.__init__ (self, robot, name)
    self._ref_traj = refTraj;
    # mask over the desired euclidian axis
    self._mask = (np.ones(3)).astype(bool);
    self._data = robot.model.createData();

  @property
  def dim(self):
    return self._mask.sum ()

  def mask(self, mask):
    assert len(mask) == 3, "The mask must have {} elements".format(3)
    self._mask = mask.astype(bool)

  def setTrajectory(self, traj):
    self._ref_traj = traj
  
  def dyn_value(self, t, q, v, update_geometry=False):
    J_am = self.robot.momentumJacobian(q, v, update_geometry);
    J_am = J_am[3:,:]
    L = J_am * v

    L_ref, dL_ref, ddL_ref = self._ref_traj(t)
    dL_des = dL_ref - self.kp * (L - L_ref)
      
    # Justin's code to compute drift (does not work)
    #g = self.robot.bias(q,0*v)
    #b = self.robot.bias(q,v)
    #b -= g;
    #com_p = self.robot.com(q)
    #cXi = SE3.Identity()
    #oXi = self.robot.data.oMi[1]
    #cXi.rotation = oXi.rotation
    #cXi.translation = oXi.translation - com_p
    #b_com = cXi.actInv(se3.Force(b[:6,0]))
    #b_com = b_com.angular;
    
    # Del Prete's quick and dirty way to compute drift
    #b_com[:] = 0.0;
    # compute momentum Jacobian at next time step assuming zero acc
    dt = 1e-3;
    q_next = se3.integrate(self.robot.model, q, dt*v);
    se3.ccrba(self.robot.model, self._data, q_next, v);
    J_am_next =  self._data.Ag[3:,:];
    b_com = (J_am_next - J_am)*v/dt;
    
    return J_am[self._mask,:], b_com[self._mask,:], dL_des[self._mask,0]


class ConfigTask(Task):

  def __init__ (self, robot, dof, ref_traj, name = "Config Task"):
    Task.__init__ (self, robot, name)

    # mask over the desired euclidian axis
    self.dim = len(dof)
    self._mask = (np.ones(self.dim)).astype(bool)
    self.dof = dof

    # desired postural configuration
    self.ref_traj = ref_traj

    # Init
    self.__error_value = np.matrix(np.empty([robot.nv,1]))
    self.__jacobian_value = np.matrix(np.identity(robot.nv))[np.array(dof)-1,:]
    self.__gain_vector = np.matrix(np.ones([1.,robot.nv]))

  @property
  def dim(self):
    return self._mask.sum ()

  def setTraj(self, ref_traj):
    self.ref_traj = ref_traj

  def setGain(self, gain_vector):
    assert gain_vector.shape == (1, self.robot.nv) 
    self.__gain_vector = np.matrix(gain_vector)

  def getGain(self):
    return self.__gain_vector

  def mask(self, mask):
    assert len(mask) == self.dim, "The mask must have {} elements".format(self.dim)
    self._mask = mask.astype(bool)
  
  def error_kin(self, t, q):
    nv = self.robot.nv
    M_ff = XYZQUATToSe3(q[:7])
    M_ff_des = XYZQUATToSe3(self.q_posture_des[:7])

    error_ff = errorInSE3(M_ff, M_ff_des).vector() 
    
    # Compute error
    error_value = self.__error_value
    error_value[:6,0] = error_ff
    error_value[6:,0] = q[7:,0] - self.q_posture_des[7:,0]

    return self.__error_value[self._mask], 0.

  def error_dyn(self, t, q, v):
    q_ref, v_ref, a_ref = self.ref_traj(t) 
    
    # Compute error
    error_value = q[self.dof,0] - q_ref
    v_err = v[np.array(self.dof)-1,0] - v_ref
    a_tot = a_ref
    
    return error_value[self._mask,0], v_err[self._mask,0], a_tot[self._mask,0]
  
  def dyn_value(self, t, q, v, update_geometry = False):
    q_ref, v_ref, a_ref = self.ref_traj(t) 
    
    # Compute error
    error_value = q[self.dof,0] - q_ref
    v_err = v[np.array(self.dof)-1,0] - v_ref
    
    J = self.__jacobian_value[self._mask,:]
    a_des = -(self.kp * error_value + self.kv * v_err) + a_ref
    drift = 0*a_des
    
    return J[self._mask,:], drift[self._mask], a_des[self._mask]

  def jacobian(self, q):
    #self.robot.mass(q)
    #M = self.robot.data.M
    #P = np.diag(np.diag(M.A)) 
    #self.__jacobian_value = np.diag(self.__gain_vector.A.squeeze())
    return self.__jacobian_value[self._mask,:] 
    #return P[self._mask,:]

