import numpy as np
import numpy.matlib
from numpy.polynomial.polynomial import polyval
#from  numpy import polyder
from  numpy.linalg import pinv
from pinocchio import SE3, log3, exp3, Motion
from derivative_filters import computeSecondOrderPolynomialFitting
import copy

def norm(v1, v2):
  return np.linalg.norm(v2.T-v1.T)

def norm(v):
  return np.linalg.norm(v.T)

def polyder(coeffs):
  return np.polyder(coeffs[::-1])[::-1]

''' Base class for a trajectory '''
class RefTrajectory (object):

  def __init__ (self, name):
    self._name = name
    self._dim = 0

  @property
  def dim(self):
    return self._dim

  def __call__ (self, t):
    return np.matrix ([]).reshape (0, 0)
    
''' An se3 trajectory with constant state and zero velocity/acceleration. '''
class ConstantSE3Trajectory (object):

  def __init__ (self, name, Mref):
    self._name = name
    self._dim = 6
    self._Mref = Mref;

  @property
  def dim(self):
    return self._dim
    
  def setReference(self, Mref):
    self._Mref = Mref;

  def __call__ (self, t):
    return (self._Mref, Motion.Zero(), Motion.Zero());
    
''' An Nd trajectory with constant state and zero velocity/acceleration. '''
class ConstantNdTrajectory (object):

  def __init__ (self, name, x_ref):
    self._name = name
    self._dim = x_ref.shape[0]
    self._x_ref = np.matrix.copy(x_ref);
    self._v_ref = np.matlib.zeros(x_ref.shape);
    self._a_ref = np.matlib.zeros(x_ref.shape);

  @property
  def dim(self):
    return self._dim
    
  def setReference(self, x_ref):
    assert x_ref.shape[0]==self._x_ref.shape[0]
    self._x_ref = x_ref;

  def __call__ (self, t):
    return (self._x_ref, self._v_ref, self._a_ref);
   
''' An Nd trajectory computed from a specified discrete-time trajectory
    by applying a polynomial fitting. 
''' 
class SmoothedNdTrajectory (object):

  ''' Constructor.
      @param x_ref A NxT numpy matrix, where N is the size of the signal and T is the number of time steps
      @param dt The time step duration in seconds
      @param window_length An odd positive integer representing the size of the window used for the polynomial fitting
  '''
  def __init__ (self, name, x_ref, dt, window_length):
    self._name = name
    self._dim = x_ref.shape[0]
    self._dt  = dt;
    (self._x_ref, self._v_ref, self._a_ref) = computeSecondOrderPolynomialFitting(x_ref, dt, window_length);

  @property
  def dim(self):
    return self._dim

  def __call__ (self, t):
    assert t>=0.0, "Time must be non-negative"
    i = int(t/self._dt);
    if(i>=self._x_ref.shape[1]):
       raise ValueError("Specified time exceeds the duration of the trajectory: "+str(t));
    return (self._x_ref[:,i], self._v_ref[:,i], self._a_ref[:,i]);

''' An SE3 trajectory computed from a specified discrete-time trajectory
    by applying a polynomial fitting. 
''' 
class SmoothedSE3Trajectory (object):

  ''' Constructor.
      @param x_ref A list of T pinocchio.SE3 objects, where T is the number of time steps
      @param dt The time step duration in seconds
      @param window_length An odd positive integer representing the size of the window used for the polynomial fitting
  '''
  def __init__ (self, name, M_ref, dt, window_length):
    self._name = name;
    self._dim = 6;
    self._dt  = dt;
    self._M_ref = M_ref;
    x_ref = np.hstack([M.translation for M in M_ref]);
    (self._x_ref, self._v_ref, self._a_ref) = computeSecondOrderPolynomialFitting(x_ref, dt, window_length);

  @property
  def dim(self):
    return self._dim

  def __call__ (self, t):
    assert t>=0.0, "Time must be non-negative"
    i = int(t/self._dt);
    if(i>=self._x_ref.shape[1]):
       raise ValueError("Specified time exceeds the duration of the trajectory: "+str(t));
    M = self._M_ref[i];
    M.translation = self._x_ref[:,i];
    v = Motion.Zero();
    a = Motion.Zero();
    v.linear = self._v_ref[:,i];
    a.linear = self._a_ref[:,i];
    return (M, v, a);

class DifferentiableEuclidianTrajectory(RefTrajectory):

  def __init__(self, name="Differentiable trajectory"):
    self.name = name
    self._time_i = []
    self._t0_l = []
    self._coeffs_l = []
    self._dcoeffs_l = []

  @property
  def dim(self):
    return self._dim

  def computeFromPoints(self, timeline, p_points, v_points):
    m_p, n_p = p_points.shape
    m_v, n_v = v_points.shape

    timeline = timeline.A.squeeze()
    N = len(timeline.T)

    assert n_p == N and n_v == N
    assert m_p == m_v

    self._dim = m_p

    # Compute time intervals
    self._time_i = []
    self._t0_l = timeline[:-1]


    for k in range(N-1):
      self._time_i.append((timeline[k], timeline[k+1]))

    # Compute the polynomial coeff on each intervals  
    self._coeffs_l = []
    self._dcoeffs_l = []

    for k in range(N-1):

      coeffs = []
      dcoeffs = []
      for i in range(self._dim):
        X0 = p_points[i,k]
        X1 = p_points[i,k+1]
        coeffs.append([X0, X1 - X0])

        dX0 = v_points[i,k]
        dX1 = v_points[i,k+1]
        dcoeffs.append([dX0, dX1 - dX0])

      self._coeffs_l.append(coeffs)
      self._dcoeffs_l.append(dcoeffs)

  def __call__ (self, t):
    #assert t <= self._time_i[-1][1], "t must be lower than the final time tf={}".format(self.time_i[-1][1])
    index = len(self._t0_l)-1
    if t > self._time_i[-1][1]:
      t = self._time_i[-1][1]
    elif t < self._time_i[0][0]:
      t = self._time_i[0][0]
      index = 0
    else:
      for k in range(len(self._t0_l)):
        if self._t0_l[k] > t:
          index = k-1 
          break

    coeffs_l = self._coeffs_l[index] 
    dcoeffs_l = self._dcoeffs_l[index] 

    t0 = self._time_i[index][0]
    t1 = self._time_i[index][1]

    tau = (t-t0)/(t1-t0)

    dim = self._dim
    X = np.matrix(np.zeros([dim,1]))
    Xd = np.matrix(np.zeros([dim,1]))

    for k in range(dim):
      if not coeffs_l[k] == []:  
        X[k] = polyval(tau,coeffs_l[k])

      if not dcoeffs_l[k] == []:  
        Xd[k] = polyval(tau,dcoeffs_l[k])

    return X, Xd
 
class TwiceDifferentiableEuclidianTrajectory(RefTrajectory):

  def __init__(self, name="Twice differentiable trajectory"):
    self.name = name
    self._time_i = []
    self._t0_l = []
    self._coeffs_l = []
    self._dcoeffs_l = []
    self._ddcoeffs_l = []

  @property
  def dim(self):
    return self._dim

  def computeFromPoints(self, timeline, p_points, v_points, a_points):
    m_p, n_p = p_points.shape
    m_v, n_v = v_points.shape
    m_a, n_a = a_points.shape

    timeline = timeline.A.squeeze()
    N = len(timeline.T)

    assert n_p == N and n_v == N and n_a == N
    assert m_p == m_v and m_p == m_a

    self._dim = m_p

    # Compute time intervals
    self._time_i = []
    self._t0_l = timeline[:-1]


    for k in range(N-1):
      self._time_i.append((timeline[k], timeline[k+1]))

    # Compute the polynomial coeff on each intervals  
    self._coeffs_l = []
    self._dcoeffs_l = []
    self._ddcoeffs_l = []

    for k in range(N-1):

      coeffs = []
      dcoeffs = []
      ddcoeffs = []
      for i in range(self._dim):
        X0 = p_points[i,k]
        X1 = p_points[i,k+1]
        coeffs.append([X0, X1 - X0])

        dX0 = v_points[i,k]
        dX1 = v_points[i,k+1]
        dcoeffs.append([dX0, dX1 - dX0])

        ddX0 = a_points[i,k]
        ddX1 = a_points[i,k+1]
        ddcoeffs.append([ddX0, ddX1 - ddX0])

      self._coeffs_l.append(coeffs)
      self._dcoeffs_l.append(dcoeffs)
      self._ddcoeffs_l.append(ddcoeffs)

  def __call__ (self, t):
    #assert t <= self._time_i[-1][1], "t must be lower than the final time tf={}".format(self.time_i[-1][1])
    index = len(self._t0_l)-1
    if t > self._time_i[-1][1]:
      t = self._time_i[-1][1]
    elif t < self._time_i[0][0]:
      t = self._time_i[0][0]
      index = 0
    else:
      for k in range(len(self._t0_l)):
        if self._t0_l[k] > t:
          index = k-1 
          break

    coeffs_l = self._coeffs_l[index] 
    dcoeffs_l = self._dcoeffs_l[index] 
    ddcoeffs_l = self._ddcoeffs_l[index] 

    t0 = self._time_i[index][0]
    t1 = self._time_i[index][1]

    tau = (t-t0)/(t1-t0)

    dim = self._dim
    X = np.matrix(np.zeros([dim,1]))
    Xd = np.matrix(np.zeros([dim,1]))
    Xdd = np.matrix(np.zeros([dim,1]))

    for k in range(dim):
      if not coeffs_l[k] == []:  
        X[k] = polyval(tau,coeffs_l[k])

      if not dcoeffs_l[k] == []:  
        Xd[k] = polyval(tau,dcoeffs_l[k])

      if not ddcoeffs_l[k] == []:  
        Xdd[k] = polyval(tau,ddcoeffs_l[k])

    return X, Xd, Xdd
      
class ZmpRefTrajectory (RefTrajectory):
  def __init__ (self, time_intervals, r_foot_p, l_foot_p, name = "ZMP ref trajectory"):
    RefTrajectory.__init__ (self,name)
    self.time_i = time_intervals
    self.r_foot_p = r_foot_p
    self.l_foot_p = l_foot_p

    self.__compute()
    self.t0_l = []
    for k in range(len(time_intervals)):
      self.t0_l.append(time_intervals[k][0])

  def __compute(self):
    self.polycoeff_l = []
    self.dpolycoeff_l = []
    num_intervals = len(self.time_i)
    print "num_intervals : ", num_intervals

    r_foot_p = self.r_foot_p
    l_foot_p = self.l_foot_p

    X0 = 0.5 * (np.array(r_foot_p[0][0]) + np.array(l_foot_p[0][0]))
    if r_foot_p[1][0] == r_foot_p[1][1]:
      X1 = r_foot_p[1][0]
    else:
      X1 = l_foot_p[1][0]

    xy_polycoeff = []
    xy_polycoeff.append([X0[0], X1[0] - X0[0]])
    xy_polycoeff.append([X0[1], X1[1] - X0[1]])
    self.polycoeff_l.append(xy_polycoeff)
    
    dxy_polycoeff = []
    dxy_polycoeff.append([X1[0] - X0[0]])
    dxy_polycoeff.append([X1[1] - X0[1]])
    self.dpolycoeff_l.append(dxy_polycoeff)

    X0 = X1

    for k in range(1,num_intervals-1):
      xy_polycoeff = []
      dxy_polycoeff = []
      if k%2:
        # SS phase
        xy_polycoeff.append([X0[0]])
        xy_polycoeff.append([X0[1]])

        dxy_polycoeff.append([])
        dxy_polycoeff.append([])

        X1 = X0
      else:
        # DS phase
        if r_foot_p[k-1][0] == r_foot_p[k-1][1]:
          X1 = l_foot_p[k][1]
        else:
          X1 = r_foot_p[k][1]

        xy_polycoeff.append([X0[0], X1[0] - X0[0]])
        xy_polycoeff.append([X0[1], X1[1] - X0[1]])
        
        dxy_polycoeff.append([X1[0] - X0[0]])
        dxy_polycoeff.append([X1[1] - X0[1]])
      
      X0 = X1
      self.polycoeff_l.append(xy_polycoeff)
      self.dpolycoeff_l.append(dxy_polycoeff)

    X1 = 0.5 * (np.array(self.r_foot_p[-1][1]) + np.array(self.l_foot_p[-1][1]))

    xy_polycoeff = []
    xy_polycoeff.append([X0[0], X1[0] - X0[0]])
    xy_polycoeff.append([X0[1], X1[1] - X0[1]])
    self.polycoeff_l.append(xy_polycoeff)

    dxy_polycoeff = []
    dxy_polycoeff.append([X1[0] - X0[0]])
    dxy_polycoeff.append([X1[1] - X0[1]])
    self.dpolycoeff_l.append(dxy_polycoeff)

  def __call__ (self, t):
    assert t <= self.time_i[-1][1], "t must be lower than the final time tf={}".format(self.time_i[-1][1])
    #index = [i for i,v in enumerate(self.t0_l) if v > t][0] - 1
    index = len(self.t0_l)-1
    if t < self.t0_l[0]:
      t = 0.
      index = 0
    else:
      for k in range(len(self.t0_l)):
        if self.t0_l[k] > t:
          index = k-1 
          break
    xy_polycoeff = self.polycoeff_l[index] 
    dxy_polycoeff = self.dpolycoeff_l[index] 

    t0 = self.time_i[index][0]
    t1 = self.time_i[index][1]

    tau = (t-t0)/(t1-t0)
    dtau_dt = 1./(t1-t0)

    # Evaluate X
    x = polyval(tau, xy_polycoeff[0])
    if len(dxy_polycoeff[0]) > 0:
      x_dot = polyval(tau, dxy_polycoeff[0]) * dtau_dt
    else:
      x_dot = 0.

    # Evaluate Y
    y = polyval(tau, xy_polycoeff[1])
    if len(dxy_polycoeff[1]) > 0:
      y_dot = polyval(tau, dxy_polycoeff[1]) * dtau_dt
    else:
      y_dot = 0.

    return np.array([x, y, 0.]).T, np.array([x_dot, y_dot, 0.]).T, np.zeros(3).T

class VerticalCoMTrajectory (RefTrajectory):
  def __init__ (self, time_intervals, x0, a_min, a_max, name="Sinusoidal trajectory"):
    RefTrajectory.__init__ (self,name)
    self.time_intervals = time_intervals
    self.x0 = x0
    self.a_min = a_min
    self.a_max = a_max
    self._dim = 3

    num_intervals = len(self.time_intervals)
    self.t0_l = []
    for k in range(num_intervals):
      self.t0_l.append(self.time_intervals[k][0])

    self.__compute()

  def __compute(self):
    self.polycoeff_l = []
    self.dpolycoeff_l = []
    self.ddpolycoeff_l = []

    num_intervals = len(self.time_intervals)

    for k in range(num_intervals):
      nx = 8 # number of coefficients for polynome on x
      x0 = self.x0
      x1 = self.x0

      t0 = self.time_intervals[k][0]
      t1 = self.time_intervals[k][1]

      dtau_dt = 1./(t1 - t0)
      if k == 0:
        tc_middle = 0.5 * (self.time_intervals[k][0] + self.time_intervals[k][1])
        tn_middle = 0.5 * (self.time_intervals[k+1][0] + self.time_intervals[k+1][1])
        x0_dot = 0.
        x1_dot = (self.a_max - self.a_min) / (tn_middle - tc_middle) 
        x1_2 = self.x0 + self.a_min
      elif k == num_intervals-1:
        tc_middle = 0.5 * (self.time_intervals[k][0] + self.time_intervals[k][1])
        tp_middle = 0.5 * (self.time_intervals[k-1][0] + self.time_intervals[k-1][1])
        x0_dot = (self.a_min - self.a_max) / (tc_middle - tp_middle) 
        x1_dot = 0.
        x1_2 = self.x0 + self.a_min
      else:
        tp_middle = 0.5 * (self.time_intervals[k-1][0] + self.time_intervals[k-1][1])
        tc_middle = 0.5 * (self.time_intervals[k][0] + self.time_intervals[k][1])
        tn_middle = 0.5 * (self.time_intervals[k+1][0] + self.time_intervals[k+1][1])

        
        if k%2: # Single support
          x0_dot = (self.a_max - self.a_min) / (tc_middle - tp_middle) 
          x1_dot = (self.a_min - self.a_max) / (tn_middle - tc_middle) 

          x1_2 = self.x0 + self.a_max
        else:
          x0_dot = (self.a_min - self.a_max) / (tc_middle - tp_middle) 
          x1_dot = (self.a_max - self.a_min) / (tn_middle - tc_middle) 

          x1_2 = self.x0 + self.a_min
          

      if not k%2:
        print k," x1_dot = ", x1_dot
      else:
        print k," x0_dot = ", x0_dot
      
      P = np.zeros([nx,nx])
      # Position
      P[0,0] = 1.
      P[1,:] += 1.
      # Velocity
      P[2,1] = 1.
      P[3,1:nx] = range(1,nx)
      # Mid trajectory constraint
      P[4,:] = np.power(0.5, range(nx))
      P[5,1:] = range(1,nx) * np.power(0.5, range(nx-1))
      # Acceleration constraint
      P[6,2] = 2.
      P[7,2:] += np.array(range(2,nx)) * np.array(range(1,nx-1))

      b = np.array([x0, x1, x0_dot, x1_dot, x1_2, 0., 0., 0.])
      x_coeff = pinv(P).dot(b)
      self.polycoeff_l.append(x_coeff)

      dx_coeff = polyder(x_coeff)
      self.dpolycoeff_l.append(dx_coeff)

      ddx_coeff = polyder(dx_coeff)
      self.ddpolycoeff_l.append(ddx_coeff)

  def __call__ (self, t):

    index = len(self.t0_l)-1
    if t > self.time_intervals[-1][1]:
      t = self.time_intervals[-1][1]
    elif t < self.time_intervals[0][0]:
      t = self.time_intervals[0][0]
      index = 0
    else:
      for k in range(len(self.t0_l)):
        if self.t0_l[k] > t:
          index = k-1 
          break

    t0 = self.time_intervals[index][0]
    t1 = self.time_intervals[index][1]

    if t0 == t1:
      tau = 0.
      dtau_dt = 0.
    else:
      tau = (t-t0)/(t1-t0)
      dtau_dt = 1./(t1-t0)
      dtau_dt = 1
    
    # Evaluate X
    x = polyval(tau, self.polycoeff_l[index])
    v = polyval(tau, self.dpolycoeff_l[index]) * dtau_dt
    a = polyval(tau, self.ddpolycoeff_l[index]) * dtau_dt ** 2

    return np.vstack([np.zeros([2,1]),x]), np.vstack([np.zeros([2,1]),v]), np.vstack([np.zeros([2,1]),a])

class FootRefTrajectory (RefTrajectory):
  def __init__ (self, time_intervals, foot_placements, z_amplitude=0.05, name="Foot ref trajectory"):
    RefTrajectory.__init__ (self,name)
    self.time_intervals = time_intervals
    self.foot_placements = foot_placements
    self.z_amplitude = z_amplitude
    self._R = np.identity(3) 

    self.__compute()
    self.t0_l = []
    for k in range(len(time_intervals)):
      self.t0_l.append(time_intervals[k][0])

  def setOrientation(self, R):
    self._R = R 

  def __compute(self):
    self.polycoeff_l = []
    self.dpolycoeff_l = []
    self.ddpolycoeff_l = []
    num_intervals = len(self.time_intervals)

    for k in range(num_intervals):
      xyz_polycoeff = []
      dxyz_polycoeff = []
      ddxyz_polycoeff = []

      foot_end_positions = self.foot_placements[k]
      if foot_end_positions[0] == foot_end_positions[1]:
        P0 = foot_end_positions.translation;
        xyz_polycoeff.append(P0[0])
        xyz_polycoeff.append(P0[1])
        xyz_polycoeff.append(P0[2])

        dxyz_polycoeff.append([])
        dxyz_polycoeff.append([])
        dxyz_polycoeff.append([])

        ddxyz_polycoeff.append([])
        ddxyz_polycoeff.append([])
        ddxyz_polycoeff.append([])
      else:
        # X trajectory
        x0 = foot_end_positions[0][0]
        x1 = foot_end_positions[1][0]
            
        nx = 6
        Px = np.zeros([nx,nx])
        Px[0,0] = 1.
        Px[1,:] += 1.
        Px[2,1] = 1.
        Px[3,1:] = range(1,nx)
        Px[4,2] = 1.
        Px[5,2:] = np.array(range(2,nx)) * np.array(range(1, nx-1))

        bx = np.array([x0, x1, 0., 0., 0., 0.])
        x_coeff = pinv(Px).dot(bx)
        xyz_polycoeff.append(x_coeff)

        dx_coeff = polyder(x_coeff)
        dxyz_polycoeff.append(dx_coeff)

        ddx_coeff = polyder(dx_coeff)
        ddxyz_polycoeff.append(ddx_coeff)

        # Y trajectory
        y0 = foot_end_positions[0][1]
        y1 = foot_end_positions[1][1]
        assert y0 == y1, "Not yet implemented case where foot moves along Y axis"
        if y0 == y1:
          xyz_polycoeff.append(y0)
          dxyz_polycoeff.append([])
          ddxyz_polycoeff.append([])

        # Z trajectory depends directly on X not on time
        z0 = foot_end_positions[0][2]
        z1 = foot_end_positions[1][2]

        nz = 7 # number of coefficients for polynome on z
        Pz = np.zeros([nz,nz])
        # Position
        Pz[0,0] = 1.
        Pz[1,:] += 1.
        # Velocity
        Pz[2,1] = 1.
        Pz[3,1:nz] = range(1,nz)
        # Mid trajectory constraint
        t_max = 0.4
        t_max = 0.5
        Pz[4,:] = np.power(t_max, range(nz))
        Pz[5,1:] = range(1,nz) * np.power(t_max, range(nz-1))
        Pz[6,2:] = np.array(range(2,nz)) * np.array(range(1,nz-1)) * np.power(t_max, range(nz-2))

        bz = np.array([z0, z1, 2., -0., 0.5*(z0+z1) + self.z_amplitude, 0., -0.1])
        bz = np.array([z0, z1, 1., -0.8, 0.5*(z0+z1) + self.z_amplitude, 0., -0.1])
        z_coeff = pinv(Pz).dot(bz)
        xyz_polycoeff.append(z_coeff)

        dz_coeff = polyder(z_coeff)
        dxyz_polycoeff.append(dz_coeff)

        ddz_coeff = polyder(dz_coeff)
        ddxyz_polycoeff.append(ddz_coeff)

      self.polycoeff_l.append(xyz_polycoeff)
      self.dpolycoeff_l.append(dxyz_polycoeff)
      self.ddpolycoeff_l.append(ddxyz_polycoeff)

  def __call__ (self, t):
    #assert t <= self.time_intervals[-1][1], "t must be lower than the final time tf={}".format(self.time_intervals[-1][1])

    index = len(self.t0_l)-1
    if t > self.time_intervals[-1][1]:
      t = self.time_intervals[-1][1]
    elif t < self.time_intervals[0][0]:
      t = self.time_intervals[0][0]
      index = 0
    else:
      for k in range(len(self.t0_l)):
        if self.t0_l[k] > t:
          index = k-1 
          break

    xyz_polycoeff = self.polycoeff_l[index] 
    dxyz_polycoeff = self.dpolycoeff_l[index] 
    ddxyz_polycoeff = self.ddpolycoeff_l[index] 

    t0 = self.time_intervals[index][0]
    t1 = self.time_intervals[index][1]

    if t0 == t1:
      tau = 0.
      dtau_dt = 0.
    else:
      tau = (t-t0)/(t1-t0)
      dtau_dt = 1.

    # Evaluate X
    x = polyval(tau, xyz_polycoeff[0])
    if len(dxyz_polycoeff[0]):
      x_dot = polyval(tau, dxyz_polycoeff[0]) * dtau_dt
    else:
      x_dot = 0.

    if len(ddxyz_polycoeff[0]):
      x_dotdot = polyval(tau, ddxyz_polycoeff[0]) * dtau_dt ** 2
    else:
      x_dotdot = 0.

    # Evaluate Y
    y = polyval(tau, xyz_polycoeff[1])
    if len(dxyz_polycoeff[1]):
      y_dot = polyval(tau, dxyz_polycoeff[1]) * dtau_dt
    else:
      y_dot = 0.
    
    if len(ddxyz_polycoeff[1]):
      y_dotdot = polyval(tau, ddxyz_polycoeff[1]) * dtau_dt ** 2 
    else:
      y_dotdot = 0.

    # Evaluate Z
    x0 = polyval(0., xyz_polycoeff[0])
    x1 = polyval(1., xyz_polycoeff[0])
    if x0 == x1:
      tau_x = 0.
      dtau_x_dt = 0.
    else:
      tau_x = (x - x0) / (x1 - x0)
      dtau_x_dt = x_dot

    z = polyval(tau_x, xyz_polycoeff[2])
    if len(dxyz_polycoeff[2]):
      z_dot = polyval(tau_x, dxyz_polycoeff[2]) * x_dot
    else:
      z_dot = 0.

    if len(ddxyz_polycoeff[2]):
      z_dotdot = polyval(tau_x, ddxyz_polycoeff[2]) * x_dot ** 2 + polyval(tau_x, dxyz_polycoeff[2]) * x_dotdot
    else:
      z_dotdot = 0.

    M = SE3.Identity() 
    v = Motion.Zero()
    a = Motion.Zero()

    M.translation = np.matrix([x, y, z]).T 
    M.rotation = self._R
    v.linear = np.matrix([x_dot, y_dot, z_dot]).T
    a.linear = np.matrix([x_dotdot, y_dotdot, z_dotdot]).T

    return M, v, a

class EndEffectorTrajectory (RefTrajectory):
  def __init__ (self, time_intervals, foot_placements, amplitude, parameters, name="End effector ref trajectory"):
    RefTrajectory.__init__ (self,name)
    self.time_intervals = time_intervals
    self.foot_placements = foot_placements
    self.amplitude = amplitude
    self.parameters = parameters
    self._R = np.identity(3) 

    self.__compute()
    self.t0_l = []
    for k in range(len(time_intervals)):
      self.t0_l.append(time_intervals[k][0])

  def setOrientation(self, R):
    self._R = R 
    #self.__compute()

  def __compute(self):
    self.polycoeff_l = []
    self.dpolycoeff_l = []
    self.ddpolycoeff_l = []
    self._R_local = []
    num_intervals = len(self.time_intervals)

    for k in range(num_intervals):
      xyz_polycoeff = []
      dxyz_polycoeff = []
      ddxyz_polycoeff = []

      foot_end_positions = self.foot_placements[k]
      P0 = np.matrix(foot_end_positions[0].translation)
      P1 = np.matrix(foot_end_positions[1].translation)


      if foot_end_positions[0] == foot_end_positions[1]:
        xyz_polycoeff.append(P0[0])
        xyz_polycoeff.append(P0[1])
        xyz_polycoeff.append(P0[2])

        dxyz_polycoeff.append([])
        dxyz_polycoeff.append([])
        dxyz_polycoeff.append([])

        ddxyz_polycoeff.append([])
        ddxyz_polycoeff.append([])
        ddxyz_polycoeff.append([])
        self._R_local.append(np.matrix(np.identity(3)))
      else:
        direction = P1 - P0
        direction_n = direction / np.linalg.norm(direction)
        X_direction = np.matrix([1.,0.,0]).T
        R = computeSO3FromVectors(X_direction, direction) # Rotation from world to local

        self._R_local.append(np.matrix(R))

        P0_local = R * P0
        P1_local = R * P1

        # X trajectory
        x0 = P0_local[0,0]
        x1 = P1_local[0,0]
            
        nx = 6
        Px = np.zeros([nx,nx])
        # Position
        Px[0,0] = 1.
        Px[1,:] += 1.
        # Velocity
        Px[2,1] = 1.
        Px[3,1:] = range(1,nx)
        # Acceleration
        Px[4,2] = 1.
        Px[5,2:] = np.array(range(2,nx)) * np.array(range(1, nx-1))

        bx = np.array([x0, x1, 0., 0., 0., 0.])
        x_coeff = pinv(Px).dot(bx)

        # Y trajectory
        y0 = P0_local[1,0]
        y1 = P1_local[1,0]

        ny = 6
        Py = np.zeros([ny,ny])
        # Position
        Py[0,0] = 1.
        Py[1,:] += 1.
        # Velocity
        Py[2,1] = 1.
        Py[3,1:] = range(1,ny)
        # Acceleration
        Py[4,2] = 1.
        Py[5,2:] = np.array(range(2,ny)) * np.array(range(1, ny-1))

        by = np.array([y0, y1, 0., 0., 0., 0.])
        y_coeff = pinv(Py).dot(by)

        # Z trajectory depends directly on X not on time
        z0 = P0_local[2,0]
        z1 = P1_local[2,0]

        nz = 7 # number of coefficients for polynome on z
        Pz = np.zeros([nz,nz])
        # Position
        Pz[0,0] = 1.
        Pz[1,:] += 1.
        # Velocity
        Pz[2,1] = 1.
        Pz[3,1:nz] = range(1,nz)
        # Mid trajectory constraint
        t_max = self.amplitude[2][0]
        z_amp = self.amplitude[2][1]
        Pz[4,:] = np.power(t_max, range(nz))
        Pz[5,1:] = range(1,nz) * np.power(t_max, range(nz-1))
        Pz[6,2:] = np.array(range(2,nz)) * np.array(range(1,nz-1)) * np.power(t_max, range(nz-2))

        # Acceleration

        bz = np.array([z0, z1, 2., -0., 0.5*(z0+z1) + z_amp, 0., -0.1])

        z_para = self.parameters[2]
        bz = np.array([z0, z1, z_para[0], z_para[1], 0.5*(z0+z1) + z_amp, 0., z_para[2]])
        z_coeff = pinv(Pz).dot(bz)

        ## Append X, Y and Z coeffs
        #n_max = max([nx,ny,nz])
        #
        #if nx != n_max:
        #  x_coeff = np.hstack([x_coeff, np.zeros([1,n_max-nx])[0]])

        #if ny != n_max:
        #  y_coeff = np.hstack([y_coeff, np.zeros([1,n_max-ny])[0]])

        #if nz != n_max:
        #  z_coeff = np.hstack([z_coeff, np.zeros([1,n_max-nz])[0]])

        #coeffs = np.matrix(np.vstack([x_coeff, y_coeff, z_coeff]))
        #coeffs_g = R.T * coeffs
  
        #x_coeff = coeffs_g[0,:].A.squeeze()
        #y_coeff = coeffs_g[1,:].A.squeeze()
        #z_coeff = coeffs_g[2,:].A.squeeze()
        
        xyz_polycoeff.append(x_coeff)

        dx_coeff = polyder(x_coeff)
        dxyz_polycoeff.append(dx_coeff)

        ddx_coeff = polyder(dx_coeff)
        ddxyz_polycoeff.append(ddx_coeff)

        xyz_polycoeff.append(y_coeff)

        dy_coeff = polyder(y_coeff)
        dxyz_polycoeff.append(dy_coeff)

        ddy_coeff = polyder(dy_coeff)
        ddxyz_polycoeff.append(ddy_coeff)

        xyz_polycoeff.append(z_coeff)

        dz_coeff = polyder(z_coeff)
        dxyz_polycoeff.append(dz_coeff)

        ddz_coeff = polyder(dz_coeff)
        ddxyz_polycoeff.append(ddz_coeff)

      self.polycoeff_l.append(xyz_polycoeff)
      self.dpolycoeff_l.append(dxyz_polycoeff)
      self.ddpolycoeff_l.append(ddxyz_polycoeff)

  def __call__ (self, t):
    #assert t <= self.time_intervals[-1][1], "t must be lower than the final time tf={}".format(self.time_intervals[-1][1])

    index = len(self.t0_l)-1
    if t > self.time_intervals[-1][1]:
      t = self.time_intervals[-1][1]
    elif t < self.time_intervals[0][0]:
      t = self.time_intervals[0][0]
      index = 0
    else:
      for k in range(len(self.t0_l)):
        if self.t0_l[k] > t:
          index = k-1 
          break

    xyz_polycoeff = self.polycoeff_l[index] 
    dxyz_polycoeff = self.dpolycoeff_l[index] 
    ddxyz_polycoeff = self.ddpolycoeff_l[index] 

    t0 = self.time_intervals[index][0]
    t1 = self.time_intervals[index][1]

    foot_end_positions = self.foot_placements[index]
    P0 = np.matrix(foot_end_positions[0].translation).T
    P1 = np.matrix(foot_end_positions[1].translation).T
    displacements = P1[:2,:] - P0 [:2,:]

    distance = norm(displacements)

    if t0 == t1:
      tau = 0.
      dtau_dt = 0.
    else:
      tau = (t-t0)/(t1-t0)
      dtau_dt = 1.

    # Evaluate X
    x = polyval(tau, xyz_polycoeff[0])
    if len(dxyz_polycoeff[0]):
      x_dot = polyval(tau, dxyz_polycoeff[0]) * dtau_dt
    else:
      x_dot = 0.

    if len(ddxyz_polycoeff[0]):
      x_dotdot = polyval(tau, ddxyz_polycoeff[0]) * dtau_dt ** 2
    else:
      x_dotdot = 0.

    # Evaluate Y
    y = polyval(tau, xyz_polycoeff[1])
    if len(dxyz_polycoeff[1]):
      y_dot = polyval(tau, dxyz_polycoeff[1]) * dtau_dt
    else:
      y_dot = 0.
    
    if len(ddxyz_polycoeff[1]):
      y_dotdot = polyval(tau, ddxyz_polycoeff[1]) * dtau_dt ** 2 
    else:
      y_dotdot = 0.

    # Evaluate Z
    x0 = polyval(0., xyz_polycoeff[0])
    x1 = polyval(1., xyz_polycoeff[0])
    if x0 == x1:
      tau_x = 0.
      dtau_x_dt = 0.
    else:
      tau_x = (x - x0) / (x1 - x0)
      dtau_x_dt = x_dot

    z = polyval(tau_x, xyz_polycoeff[2])
    if len(dxyz_polycoeff[2]):
      z_dot = polyval(tau_x, dxyz_polycoeff[2]) * x_dot
    else:
      z_dot = 0.

    if len(ddxyz_polycoeff[2]):
      z_dotdot = polyval(tau_x, ddxyz_polycoeff[2]) * x_dot ** 2 + polyval(tau_x, dxyz_polycoeff[2]) * x_dotdot
    else:
      z_dotdot = 0.

    M = SE3.Identity() 
    v = Motion.Zero()
    a = Motion.Zero()

    R = self._R_local[index]
    M.translation = R.T * np.matrix([x, y, z]).T 
    M.rotation = self._R
    v.linear = R.T * np.matrix([x_dot, y_dot, z_dot]).T
    a.linear = R.T * np.matrix([x_dotdot, y_dotdot, z_dotdot]).T

    return M, v, a

  def plot(self):
    import matplotlib.pyplot as plt

    N = 100
    t0 = self.time_intervals[0][0]
    tf = self.time_intervals[-1][1]
    fig = plt.figure()
    t = np.linspace(t0, tf, N)

    points = np.matrix(np.empty([3,N]))
    for k in range(N):
      M,_,_ = self(t[k])
      points[:,k] = M.translation

    plt.subplot(311)
    plt.plot(t,points[0,:].A.squeeze()) 

    plt.subplot(312)
    plt.plot(t,points[1,:].A.squeeze()) 

    plt.subplot(313)
    plt.plot(t,points[2,:].A.squeeze()) 

#from bezier_curves import BezierCurves
#
#class EndEffectorBezierTrajectory (RefTrajectory):
#  def __init__ (self, time_intervals, curves_l, end_effector_orientation_l, name="Bezier ref trajectory"):
#    for k in range(len(curves_l)):
#      assert curves_l[k].dim == 3., "Bezier curves must be of dimension 3"
#
#    RefTrajectory.__init__ (self,name)
#    self.time_intervals = time_intervals
#    self.curves_l = curves_l
#    self.orientation_l = end_effector_orientation_l
#    self._R = np.identity(3) 
#
#    ## Compute derivates
#    self.dcurves_l = []
#    self.ddcurves_l = []
#    for k in range(len(self.curves_l)):
#      self.dcurves_l.append(self.curves_l[k].derived())
#      self.ddcurves_l.append(self.dcurves_l[k].derived())
#
#    self.t0_l = []
#    for k in range(len(time_intervals)):
#      self.t0_l.append(time_intervals[k][0])
#
#  def setOrientation(self, R):
#    self._R = R 
#
#  def __call__ (self, t):
#    #assert t <= self.time_intervals[-1][1], "t must be lower than the final time tf={}".format(self.time_intervals[-1][1])
#
#    index = len(self.t0_l)-1
#    if t > self.time_intervals[-1][1]:
#      t = self.time_intervals[-1][1]
#    elif t < self.time_intervals[0][0]:
#      t = self.time_intervals[0][0]
#      index = 0
#    else:
#      for k in range(len(self.t0_l)):
#        if self.t0_l[k] > t:
#          index = k-1 
#          break
#
#    curve = self.curves_l[index]
#    dcurve = self.dcurves_l[index]
#    ddcurve = self.ddcurves_l[index]
#
#    t0 = self.time_intervals[index][0]
#    t1 = self.time_intervals[index][1]
#
#    if t0 == t1:
#      tau = 0.
#      dtau_dt = 0.
#    else:
#      tau = (t-t0)/(t1-t0)
#      dtau_dt = 1./(t1-t0)
#
#    # Evaluate curves
#    pos = curve(tau)
#    vel = dcurve(tau) * dtau_dt
#    acc = ddcurve(tau) * dtau_dt**2
#
#    M = SE3.Identity() 
#    v = Motion.Zero()
#    a = Motion.Zero()
#
#    M.translation = pos 
#    #M.rotation = self._R
#    M.rotation = self.orientation_l[index][0] * exp3(tau * self.orientation_l[index][1])
#    v.linear = vel
#    a.linear = acc
#
#    return M, v, a
