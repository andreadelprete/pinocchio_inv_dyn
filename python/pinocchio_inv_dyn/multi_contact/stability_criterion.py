# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:54:39 2016

@author: adelpret
"""

from com_acc_LP import ComAccLP
from pinocchio_inv_dyn.optimization.solver_LP_abstract import LP_status, LP_status_string
from robust_equilibrium_DLP import RobustEquilibriumDLP
import pinocchio_inv_dyn.plot_utils as plut
import matplotlib.pyplot as plt
from math import atan, pi

from pinocchio_inv_dyn.sot_utils import qpOasesSolverMsg
import numpy as np
from numpy.linalg import norm
from numpy import sqrt
import cProfile

np.set_printoptions(precision=2, suppress=True, linewidth=100);
EPS = 1e-5;

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds);

    def __str__(self, prefix=""):
        res = "";
        for (key,value) in self.__dict__.iteritems():
            if (isinstance(value, np.ndarray) and len(value.shape)==2 and value.shape[0]>value.shape[1]):
                res += prefix+" - " + key + ": " + str(value.T) + "\n";
            elif (isinstance(value, Bunch)):
                res += prefix+" - " + key + ":\n" + value.__str__(prefix+"    ") + "\n";
            else:
                res += prefix+" - " + key + ": " + str(value) + "\n";
        return res[:-1];


class StabilityCriterion(object):
    _name = ""
    _maxIter = 0;
    _verb = 0;
    _com_acc_solver = None;
    _equilibrium_solver = None;
    _c0 = None;
    _dc0 = None;
    _v = None;
    
    _computationTime = 0.0;
    _outerIterations = 0;
    _innerIterations = 0;
    
    def __init__ (self, name, c0, dc0, contact_points, contact_normals, mu, g, mass, 
                  maxIter=1000, verb=0, regularization=1e-5, solver='qpoases'):
        ''' Constructor
            @param c0 Initial CoM position
            @param dc0 Initial CoM velocity
            @param g Gravity vector
            @param regularization Weight of the force minimization, the higher this value, the sparser the solution
        '''
        assert mass>0.0, "Mass is not positive"
        assert mu>0.0, "Friction coefficient is not positive"
        assert np.asarray(c0).squeeze().shape[0]==3, "Com position vector has not size 3"
        assert np.asarray(dc0).squeeze().shape[0]==3, "Com velocity vector has not size 3"
        assert np.asarray(contact_points).shape[1]==3, "Contact points have not size 3"
        assert np.asarray(contact_normals).shape[1]==3, "Contact normals have not size 3"
        assert np.asarray(contact_points).shape[0]==np.asarray(contact_normals).shape[0], "Number of contact points do not match number of contact normals"
        self._name              = name;
        self._maxIter           = maxIter;
        self._verb              = verb;
        self._c0                = np.asarray(c0).squeeze().copy();
        self._dc0               = np.asarray(dc0).squeeze().copy();
        self._mass              = mass;
        self._g                 = np.asarray(g).squeeze().copy();
#        self._regularization    = regularization;
        self._mu                = mu;
        self._contact_points    = np.asarray(contact_points).copy();
        self._contact_normals   = np.asarray(contact_normals).copy();
        if(norm(self._dc0)!=0.0):
            self._v = self._dc0/norm(self._dc0);
        else:
            self._v = np.array([1.0, 0.0, 0.0]);
        self._com_acc_solver  = ComAccLP(self._name+"_comAccLP", self._c0, self._v, self._contact_points, self._contact_normals, 
                                         self._mu, self._g, self._mass, maxIter, verb-1, regularization, solver);
        self._equilibrium_solver = RobustEquilibriumDLP(name+"_robEquiDLP", self._contact_points, self._contact_normals, 
                                                        self._mu, self._g, self._mass, verb=verb-1);
        

    def set_contacts(self, contact_points, contact_normals, mu):
        self._com_acc_solver.set_contacts(contact_points, contact_normals, mu);
        self._equilibrium_solver = RobustEquilibriumDLP(self._name, contact_points, contact_normals, mu, self._g, self._mass, verb=self._verb);


    def can_I_stop(self, c0=None, dc0=None, T_0=0.5, MAX_ITER=1000):
        ''' Determine whether the system can come to a stop without changing contacts.
            Keyword arguments:
              c0 -- initial CoM position 
              dc0 -- initial CoM velocity 
              contact points -- a matrix containing the contact points
              contact normals -- a matrix containing the contact normals
              mu -- friction coefficient (either a scalar or an array)
              mass -- the robot mass
              T_0 -- an initial guess for the time to stop
            Output: An object containing the following member variables:
              is_stable -- boolean value
              c -- final com position
              dc -- final com velocity
              ddc_min -- minimum feasible com acceleration at the final com position (to be used as a stability margin)
              t -- time taken to reach the final com state
              computation_time -- time taken to solve all the LPs
        '''
        #- Initialize: alpha=0, Dalpha=||dc0||
        #- LOOP:
        #    - Find min com acc for current com pos (i.e. DDalpha_min)
        #    - If DDalpha_min>0: return False
        #    - Find alpha_max (i.e. value of alpha corresponding to right vertex of active inequality)
        #    - Initialize: t=T_0, t_ub=10, t_lb=0
        #    LOOP:
        #        - Integrate LDS until t: DDalpha = d/b - (a/d)*alpha
        #        - if(Dalpha(t)==0 && alpha(t)<=alpha_max):  return (True, alpha(t)*v)
        #        - if(alpha(t)==alpha_max && Dalpha(t)>0):   alpha=alpha(t), Dalpha=Dalpha(t), break
        #        - if(alpha(t)<alpha_max && Dalpha>0):       t_lb=t, t=(t_ub+t_lb)/2
        #        - else                                      t_ub=t, t=(t_ub+t_lb)/2
        assert T_0>0.0, "Time is not positive"
        if(c0 is not None):
            assert np.asarray(c0).squeeze().shape[0]==3, "CoM has not size 3"
            self._c0 = np.asarray(c0).squeeze().copy();
        if(dc0 is not None):
            assert np.asarray(dc0).squeeze().shape[0]==3, "CoM velocity has not size 3"
            self._dc0 = np.asarray(dc0).squeeze().copy();
            if(norm(self._dc0)!=0.0):
                self._v = self._dc0/norm(self._dc0);
        if((c0 is not None) or (dc0 is not None)):
            self._com_acc_solver.set_com_state(self._c0, self._v);
            
        # Initialize: alpha=0, Dalpha=||dc0||
        t_int = 0.0;
        alpha = 0.0;
        Dalpha = norm(self._dc0);
        last_iteration = False;
        self._computationTime = 0.0;
        self._outerIterations = 0;
        self._innerIterations = 0;
        last_ddc_min = None;
        
        if(Dalpha==0.0):
            r = self._equilibrium_solver.compute_equilibrium_robustness(self._c0);
            return Bunch(is_stable=(r>=0.0), c=self._c0, dc=self._dc0, t=0.0, computation_time=0.0,
                         ddc_min=None);

        for iiii in range(MAX_ITER):
            if(last_iteration):
                if(self._verb>0):
                    print "[%s] Algorithm converged to Dalpha=%.3f" % (self._name, Dalpha);
                return Bunch(is_stable=False, c=self._c0+alpha*self._v, dc=Dalpha*self._v, t=t_int, 
                             computation_time=self._computationTime, ddc_min=0.0);

            (imode, DDalpha_min, a, alpha_min, alpha_max) = self._com_acc_solver.compute_max_deceleration(alpha);
            
            self._computationTime += self._com_acc_solver.getLpTime();
            self._outerIterations += 1;
            
            if(imode==LP_status.INFEASIBLE):
                # Linear Program was not feasible, suggesting there is no feasible CoM acceleration for the given CoM position
                if(self._verb>1):
                    from com_acc_LP import ComAccPP
                    self._comAccPP = None;
                    try:
                        self._comAccPP = ComAccPP(self._name+"_comAccPP", self._c0, self._v, self._contact_points, self._contact_normals, 
                                                  self._mu, self._g, self._mass, verb=self._verb-1);
                        (imode2, DDalpha_min2, a2, alpha_max2) = self._comAccPP.compute_max_deceleration(alpha);
                    except Exception as e:
                        print "[%s] Error while trying to compute min com acc with ComAccPP. "%(self._name)+str(e);
                    
                    if(self._comAccPP is not None and imode2==0):
                        raise ValueError("[%s] Computing min com acc with PP gave different result than with LP: "%(self._name)+str(DDalpha_min2));
                return Bunch(is_stable=False, c=self._c0+alpha*self._v, dc=Dalpha*self._v, t=t_int,
                             computation_time=self._computationTime, ddc_min=None);

            if(imode!=LP_status.OPTIMAL):
                raise ValueError("[%s] Error while solving com acc LP: %s"%(self._name, LP_status_string[imode]))
                
            last_ddc_min = DDalpha_min;

            if(self._verb>0):
                print "[%s] DDalpha_min=%.3f, alpha=%.3f, Dalpha=%.3f, alpha_max=%.3f, a=%.3f" % (self._name, DDalpha_min, alpha, Dalpha, alpha_max, a);

            if(DDalpha_min>-EPS):
                if(self._verb>0):
                    print "[%s] Minimum com acceleration is positive, so I cannot stop"%(self._name);
                return Bunch(is_stable=False, c=self._c0+alpha*self._v, dc=Dalpha*self._v, t=t_int,
                             computation_time=self._computationTime, ddc_min=DDalpha_min);

            # DDalpha_min is the acceleration corresponding to the current value of alpha
            # compute DDalpha_0, that is the acceleration corresponding to alpha=0
            DDalpha_0 = DDalpha_min - a*alpha;
            
            # If DDalpha_min is not always negative on the current segment then update 
            # alpha_max to the point where DDalpha_min becomes zero
            if( DDalpha_0 + a*alpha_max > 0.0):
                alpha_max = -DDalpha_0 / a;
                last_iteration = True;
                if(self._verb>0):
                    print "[%s] Updated alpha_max %.3f"%(self._name, alpha_max);
            
            # Initialize: t=T_0, t_ub=10, t_lb=0
            t    = T_0;
            t_ub = 10.0;    # hypothesis: 10 seconds are always sufficient to stop
            t_lb = 0.0;
            if(abs(a)<EPS):
                # if the acceleration is constant over time I can compute directly the time needed
                # to bring the velocity to zero:
                #     Dalpha(t) = Dalpha(0) + t*DDalpha = 0
                #     t = -Dalpha(0)/DDalpha
                t_zero = - Dalpha / DDalpha_min;
                alpha_t  = alpha + t_zero*Dalpha + 0.5*t_zero*t_zero*DDalpha_min;
                if(alpha_t <= alpha_max+EPS):
                    if(self._verb>0):
                        print "DDalpha_min is independent from alpha, algorithm converged to Dalpha=0";
                    return Bunch(is_stable=True, c=self._c0+alpha_t*self._v, dc=0.0*self._v, t=t_int+t_zero,
                                 computation_time=self._computationTime, ddc_min=DDalpha_min);

                # if alpha reaches alpha_max before the velocity is zero, then compute the time needed to reach alpha_max
                #     alpha(t) = alpha(0) + t*Dalpha(0) + 0.5*t*t*DDalpha = alpha_max
                #     t = (- Dalpha(0) +/- sqrt(Dalpha(0)^2 - 2*DDalpha(alpha(0)-alpha_max))) / DDalpha;
                # where DDalpha = -d[i_DDalpha_min]
                # Having two solutions, we take the smallest one because we want to find the first time
                # at which alpha reaches alpha_max
                delta = sqrt(Dalpha**2 - 2*DDalpha_min*(alpha-alpha_max))
                t = ( - Dalpha + delta) / DDalpha_min;
                if(t<0.0):
                    # If the smallest time at which alpha reaches alpha_max is negative print a WARNING because this should not happen
                    print "[%s] WARNING: Time is negative: t=%.3f, alpha=%.3f, Dalpha=%.3f, DDalpha_min=%.3f, alpha_max=%.3f"%(self._name,t,alpha,Dalpha,DDalpha_min,alpha_max);
                    t = (-Dalpha - delta) / DDalpha_min;
                    if(t<0.0):
                        # If also the largest time is negative print an ERROR and return
                        raise ValueError("[%s] ERROR: Time is still negative: t=%.3f, alpha=%.3f, Dalpha=%.3f, DDalpha_min=%.3f, alpha_max=%.3f"%(self._name,t,alpha,Dalpha,DDalpha_min,alpha_max));
            else:
                # Try to compute the time at which the velocity will be zero:
                #   Dalpha(t) = omega*sh*alpha + ch*Dalpha + omega*sh*(DDalpha_0/a) = 0
                #   omega*th*alpha + Dalpha + omega*th*(DDalpha_0/a) = 0
                #   th*omega*(alpha + DDalpha_0/a) = - Dalpha
                #   tanh(omega*t) = - Dalpha / (omega*(alpha + DDalpha_0/a))
                #   omega*t = atanh(- Dalpha / (omega*(alpha + DDalpha_0/a)))
                #   t = atanh(- Dalpha / (omega*(alpha + DDalpha_0/a))) / omega
                omega = sqrt(a+0j);
                tmp = - Dalpha / (omega*(alpha + (DDalpha_0/a)));
                if(abs(np.real(tmp))<1.0):
                    t = np.arctanh(tmp) / omega;
                    if(abs(np.imag(t)) > EPS):
                        raise ValueError("[%s] ERROR time to reach zero vel is imaginary: "%self._name +str(t)+"a=%.3f"%(a));
                    t = np.real(t);
                    t_ub = t;   # use this time as an upper bound for the search
                else:
                    # Here I already know that I won't be able to stop, but I keep integrating to find the min velocity I can reach.
                    if(self._verb>0):
                        print "[%s] INFO argument of arctanh greater than 1 (%.3f), meaning that zero vel will never be reached"%(self._name,np.real(tmp));  
                    #return Bunch(is_stable=False, c=self._c0+alpha*self._v, dc=Dalpha*self._v, t=t_int);
                
                    
            bisection_converged = False;
            for jjjj in range(MAX_ITER):
                # Integrate LDS until t: DDalpha = a*alpha - d
                if(abs(a)>EPS):
                    # if a!=0 then the acceleration is a linear function of the position and I need to use this formula to integrate
                    sh = np.sinh(omega*t);
                    ch = np.cosh(omega*t);
                    alpha_t  = ch*alpha + sh*Dalpha/omega - (1.0-ch)*(DDalpha_0/a);
                    Dalpha_t = omega*sh*alpha + ch*Dalpha + omega*sh*(DDalpha_0/a);
                else:
                    # if a==0 then the acceleration is constant and I need to use this formula to integrate
                    alpha_t  = alpha + t*Dalpha + 0.5*t*t*DDalpha_0;
                    Dalpha_t = Dalpha + t*DDalpha_0;
                
                if(np.imag(alpha_t) != 0.0):
                    raise ValueError("ERROR alpha is imaginary: "+str(alpha_t));
                if(np.imag(Dalpha_t) != 0.0):
                    raise ValueError("ERROR Dalpha is imaginary: "+str(Dalpha_t))
                    
                alpha_t = np.real(alpha_t);
                Dalpha_t = np.real(Dalpha_t);
                if(self._verb>1):
                    print "[%s] Bisection iter"%(self._name),jjjj,"alpha",alpha_t,"Dalpha",Dalpha_t,"t", t
                
                if(abs(Dalpha_t)<EPS and alpha_t <= alpha_max+EPS):
                    if(self._verb>0):
                        print "[%s] Algorithm converged to Dalpha=0"%self._name;
                    self._innerIterations += jjjj;
                    t_int += t;
                    return Bunch(is_stable=True, c=self._c0+alpha_t*self._v, dc=Dalpha_t*self._v, t=t_int,
                                 computation_time=self._computationTime, ddc_min=DDalpha_0+a*alpha_t);
                if(abs(alpha_t-alpha_max)<EPS and Dalpha_t>0):
                    alpha = alpha_max+EPS;
                    Dalpha = Dalpha_t;
                    t_int += t;
                    bisection_converged = True;
                    self._innerIterations += jjjj;
                    break;
                if(alpha_t<alpha_max and Dalpha_t>0):       
                    t_lb=t;
                    t=0.5*(t_ub+t_lb);
                else:                                      
                    t_ub=t; 
                    t=0.5*(t_ub+t_lb);
            
            if(not bisection_converged):
                raise ValueError("[%s] Bisection search did not converge in %d iterations"%(self._name, MAX_ITER));
    
        raise ValueError("[%s] Algorithm did not converge in %d iterations"%(self._name, MAX_ITER));


    def predict_future_state(self, t_pred, c0=None, dc0=None, MAX_ITER=1000):
        ''' Compute what the CoM state will be at the specified time instant if the system
            applies maximum CoM deceleration parallel to the current CoM velocity
            Keyword arguments:
            t_pred -- Future time at which the prediction is made
            c0 -- initial CoM position 
            dc0 -- initial CoM velocity 
            Output: An object with the following member variables:
            t -- time at which the integration has stopped (equal to t_pred, unless something went wrong)
            c -- final com position
            dc -- final com velocity
        '''
        #- Initialize: alpha=0, Dalpha=||dc0||
        #- LOOP:
        #    - Find min com acc for current com pos (i.e. DDalpha_min)
        #    - If DDalpha_min>0: return False
        #    - Find alpha_max (i.e. value of alpha corresponding to right vertex of active inequality)
        #    - Initialize: t=T_0, t_ub=10, t_lb=0
        #    LOOP:
        #        - Integrate LDS until t: DDalpha = d/b - (a/d)*alpha
        #        - if(Dalpha(t)==0 && alpha(t)<=alpha_max):  return (True, alpha(t)*v)
        #        - if(alpha(t)==alpha_max && Dalpha(t)>0):   alpha=alpha(t), Dalpha=Dalpha(t), break
        #        - if(alpha(t)<alpha_max && Dalpha>0):       t_lb=t, t=(t_ub+t_lb)/2
        #        - else                                      t_ub=t, t=(t_ub+t_lb)/2
        assert t_pred>0.0, "Prediction time is not positive"
        if(c0 is not None):
            assert np.asarray(c0).squeeze().shape[0]==3, "CoM position has not size 3"
            self._c0 = np.asarray(c0).squeeze().copy();
        if(dc0 is not None):
            assert np.asarray(dc0).squeeze().shape[0]==3, "CoM velocity has not size 3"
            self._dc0 = np.asarray(dc0).squeeze().copy();
            if(norm(self._dc0)!=0.0):
                self._v = self._dc0/norm(self._dc0);
        if((c0 is not None) or (dc0 is not None)):
            self._com_acc_solver.set_com_state(self._c0, self._dc0);
        
        # Initialize: alpha=0, Dalpha=||dc0||
        alpha = 0.0;
        Dalpha = norm(self._dc0);
        Dalpha_min = Dalpha;    # minimum velocity found while integrating
        self._computationTime = 0.0;
        self._outerIterations = 0;
        self._innerIterations = 0;
        t_int = 0.0;    # current time
        t_min = 0.0;    # time corresponding to minimum velocity
        min_vel_reached = False;
        
        if(Dalpha==0.0):
            r = self._equilibrium_solver.compute_equilibrium_robustness(self._c0);
            if(r>=0.0):
                return Bunch(t=t_pred, c=self._c0, dc=self._dc0, t_min=0.0, dc_min=self._dc0);
            raise ValueError("[%s] NOT IMPLEMENTED YET: initial velocity is zero but system is not in static equilibrium!"%(self._name));

        for iiii in range(MAX_ITER):
            (imode, DDalpha_min, a, alpha_min, alpha_max) = self._com_acc_solver.compute_max_deceleration(alpha);
            
            self._computationTime += self._com_acc_solver.getLpTime();
            self._outerIterations += 1;
            
            if(imode!=0):
                # Linear Program was not feasible, suggesting there is no feasible CoM acceleration for the given CoM position
                if(min_vel_reached):
                    return Bunch(t=t_int, c=self._c0+alpha*self._v, dc=Dalpha*self._v, t_min=t_min, dc_min=Dalpha_min*self._v);
                return Bunch(t=t_int, c=self._c0+alpha*self._v, dc=Dalpha*self._v, t_min=t_int, dc_min=Dalpha*self._v);

            if(self._verb>0):
                print "[%s] t=%.3f, DDalpha_min=%.3f, alpha=%.3f, Dalpha=%.3f, alpha_max=%.3f, a=%.3f" % (self._name, t_int, DDalpha_min, alpha, Dalpha, alpha_max, a);

            # DDalpha_min is the acceleration corresponding to the current value of alpha
            # compute DDalpha_0, that is the acceleration corresponding to alpha=0
            DDalpha_0 = DDalpha_min - a*alpha;
            
            if(not min_vel_reached):
                if(DDalpha_min>=0.0):
                    min_vel_reached = True;
                    Dalpha_min = Dalpha;
                    t_min = t_int;
                elif( DDalpha_0 + a*alpha_max > 0.0):
                    # If we have not found yet the point of minimum velocity,
                    # if DDalpha_min is not always negative on the current segment then update 
                    # alpha_max to the point where DDalpha_min becomes zero
                    alpha_max = -DDalpha_0 / a;
                    if(self._verb>0):
                        print "[%s] Updated alpha_max %.3f"%(self._name, alpha_max);
                        
            # Initialize initial guess and bounds on integration time
            t_ub = t_pred-t_int;
            t_lb = 0.0;
            t    = t_ub;
            if(abs(a)<EPS):
                # if the acceleration is constant over time I can compute directly the time needed
                # to bring the velocity to zero:
                #     Dalpha(t) = Dalpha(0) + t*DDalpha = 0
                #     t = -Dalpha(0)/DDalpha
                t = - Dalpha / DDalpha_min;
                alpha_t  = alpha + t*Dalpha + 0.5*t*t*DDalpha_min;
                if(alpha_t <= alpha_max+EPS):
                    if(self._verb>0):
                        print "DDalpha_min is independent from alpha, algorithm converged to Dalpha=0";
                    # hypothesis: after I reach Dalpha=0 I can maintain it (i.e. DDalpha=0 is feasible)
                    return Bunch(t=t_pred, c=self._c0+alpha_t*self._v, dc=0.0*self._v, t_min=t_int+t, dc_min=0*self._v);

                # if alpha reaches alpha_max before the velocity is zero, then compute the time needed to reach alpha_max
                #     alpha(t) = alpha(0) + t*Dalpha(0) + 0.5*t*t*DDalpha = alpha_max
                #     t = (- Dalpha(0) +/- sqrt(Dalpha(0)^2 - 2*DDalpha(alpha(0)-alpha_max))) / DDalpha;
                # where DDalpha = -d[i_DDalpha_min]
                # Having two solutions, we take the smallest one because we want to find the first time
                # at which alpha reaches alpha_max
                delta = sqrt(Dalpha**2 - 2*DDalpha_min*(alpha-alpha_max))
                t = ( - Dalpha + delta) / DDalpha_min;
                if(t<0.0):
                    raise ValueError("[%s] ERROR: Time to reach alpha_max with constant acceleration is negative: t=%.3f, alpha=%.3f, Dalpha=%.3f, DDalpha_min=%.3f, alpha_max=%.3f"%(self._name,t,alpha,Dalpha,DDalpha_min,alpha_max));
                # ensure we do not overpass the specified t_pred
                t = min([t, t_ub]);
                
            # integrate until either of these conditions is true:
            # - you reach t=t_pred (while not passing over alpha_max and maintaining Dalpha>=0)
            # - you reach alpha_max (while not passing over t_pred and maintaining Dalpha>=0)
            # - you reach Dalpha=0 (while not passing over alpha_max and t_pred)
            bisection_converged = False;
            for jjjj in range(MAX_ITER):
                # Integrate LDS until t: DDalpha = a*alpha - d
                if(abs(a)>EPS):
                    # if a!=0 then the acceleration is a linear function of the position and I need to use this formula to integrate
                    omega = sqrt(a+0j);
                    sh = np.sinh(omega*t);
                    ch = np.cosh(omega*t);
                    alpha_t  = ch*alpha + sh*Dalpha/omega - (1.0-ch)*(DDalpha_0/a);
                    Dalpha_t = omega*sh*alpha + ch*Dalpha + omega*sh*(DDalpha_0/a);
                else:
                    # if a==0 then the acceleration is constant and I need to use this formula to integrate
                    alpha_t  = alpha + t*Dalpha + 0.5*t*t*DDalpha_0;
                    Dalpha_t = Dalpha + t*DDalpha_0;
                
                if(np.imag(alpha_t) != 0.0):
                    raise ValueError("ERROR alpha is imaginary: "+str(alpha_t));
                if(np.imag(Dalpha_t) != 0.0):
                    raise ValueError("ERROR Dalpha is imaginary: "+str(Dalpha_t))
                    
                alpha_t = np.real(alpha_t);
                Dalpha_t = np.real(Dalpha_t);
                if(self._verb>1):
                    print "[%s] Bisection iter"%(self._name),jjjj,"alpha",alpha_t,"Dalpha",Dalpha_t,"t", t
                
                if(alpha_t <= alpha_max+EPS and (abs(Dalpha_t)<EPS or (Dalpha_t>-EPS and abs(t+t_int-t_pred)<EPS))):
                    # if I did not pass over alpha_max and velocity is zero OR
                    # if I did not pass over alpha_max and velocity is positive and I reached t_pred
                    if(self._verb>0):
                        print "[%s] Algorithm converged to Dalpha=%.3f"%(self._name, Dalpha_t);
                    self._innerIterations += jjjj;
                    if(min_vel_reached):
                        return Bunch(t=t_pred, c=self._c0+alpha_t*self._v, dc=Dalpha_t*self._v, t_min=t_min, dc_min=Dalpha_min*self._v);
                    return Bunch(t=t_pred, c=self._c0+alpha_t*self._v, dc=Dalpha_t*self._v, t_min=t_pred, dc_min=Dalpha_t*self._v);

                if(abs(alpha_t-alpha_max)<EPS and Dalpha_t>0):
                    alpha = alpha_max+EPS;
                    Dalpha = Dalpha_t;
                    t_int += t;
                    bisection_converged = True;
                    self._innerIterations += jjjj;
                    break;
                    
                if(alpha_t<alpha_max and Dalpha_t>0):       
                    t_lb=t;
                    t=0.5*(t_ub+t_lb);
                else:
                    t_ub=t; 
                    t=0.5*(t_ub+t_lb);
            
            if(not bisection_converged):
                raise ValueError("[%s] Bisection search did not converge in %d iterations"%(self._name, MAX_ITER));
    
        raise ValueError("[%s] Algorithm did not converge in %d iterations"%(self._name, MAX_ITER));
    
    
def test(N_CONTACTS = 2, solver='qpoases', verb=0):
    from pinocchio_inv_dyn.multi_contact.utils import generate_contacts, compute_GIWC, find_static_equilibrium_com, can_I_stop
    DO_PLOTS = True;
    PLOT_3D = False;
    mass = 75;             # mass of the robot
    mu = 0.5;           # friction coefficient
    lx = 0.1;           # half foot size in x direction
    ly = 0.07;          # half foot size in y direction
    #First, generate a contact configuration
    CONTACT_POINT_UPPER_BOUNDS = [ 0.5,  0.5,  0.5];
    CONTACT_POINT_LOWER_BOUNDS = [-0.5, -0.5,  0.0];
    gamma = atan(mu);   # half friction cone angle
    RPY_LOWER_BOUNDS = [-2*gamma, -2*gamma, -pi];
    RPY_UPPER_BOUNDS = [+2*gamma, +2*gamma, +pi];
    MIN_CONTACT_DISTANCE = 0.3;
    g_vector = np.array([0., 0., -9.81]);
    X_MARG = 0.07;
    Y_MARG = 0.07;
    
    succeeded = False;
    while(not succeeded):
        (p, N) = generate_contacts(N_CONTACTS, lx, ly, mu, CONTACT_POINT_LOWER_BOUNDS, CONTACT_POINT_UPPER_BOUNDS, 
                                   RPY_LOWER_BOUNDS, RPY_UPPER_BOUNDS, MIN_CONTACT_DISTANCE, False);
        X_LB = np.min(p[:,0]-X_MARG);
        X_UB = np.max(p[:,0]+X_MARG);
        Y_LB = np.min(p[:,1]-Y_MARG);
        Y_UB = np.max(p[:,1]+Y_MARG);
        Z_LB = np.min(p[:,2]-0.05);
        Z_UB = np.max(p[:,2]+1.5);
        (H,h) = compute_GIWC(p, N, mu, False);
        (succeeded, c0) = find_static_equilibrium_com(mass, [X_LB, Y_LB, Z_LB], [X_UB, Y_UB, Z_UB], H, h);
        
    dc0 = np.random.uniform(-1, 1, size=3); 
#    dc0[:] = 0;
    
    if(DO_PLOTS):
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
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = Axes3D(fig)
#        ax = fig.gca(projection='3d')
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
   
    if(verb>0):
        print "p:", p.T;
        print "N", N.T;
        print "c", c0.T;
        print "dc", dc0.T;
        
    stabilitySolver = StabilityCriterion("ss", c0, dc0, p, N, mu, g_vector, mass, verb=verb, solver=solver);
    try:
        res = stabilitySolver.can_I_stop();
    except Exception as e:
        print "\n *** New algorithm failed:", e,"\nRe-running the algorithm with high verbosity\n"
        stabilitySolver._verb = 1;
        try:
            res = stabilitySolver.can_I_stop();
        except Exception as e:
            raise
        
    try:
        (has_stopped2, c_final2, dc_final2) = can_I_stop(c0, dc0, p, N, mu, mass, 1.0, 100, verb=verb, DO_PLOTS=DO_PLOTS);
        if((res.is_stable != has_stopped2) or 
            not np.allclose(res.c, c_final2, atol=1e-3) or 
            not np.allclose(res.dc, dc_final2, atol=1e-3)):
            print "\nERROR: the two algorithms gave different results!"
            print "New algorithm:", res.is_stable, res.c, res.dc;
            print "Old algorithm:", has_stopped2, c_final2, dc_final2;
            print "Errors:", norm(res.c-c_final2), norm(res.dc-dc_final2), "\n";
    except Exception as e:
        print "\n\n *** Old algorithm failed: ", e
        print "Results of new algorithm is", res.is_stable, "c0", c0, "dc0", dc0, "cFinal", res.c, "dcFinal", res.dc,"\n";
        
    return (stabilitySolver._computationTime, stabilitySolver._outerIterations, stabilitySolver._innerIterations);
        

if __name__=="__main__":
    N_CONTACTS = 2;
    SOLVER = 'cvxopt'; # cvxopt
    VERB = 0;
    N_TESTS = range(0,10);
    time    = np.zeros(len(N_TESTS));
    outIter = np.zeros(len(N_TESTS));
    inIter  = np.zeros(len(N_TESTS));
    # test 392 with 2 contacts give different results
    # test 264 with 3 contacts give different results
    j = 0;
    for i in N_TESTS:
        try:
            np.random.seed(i);
            (time[j], outIter[j], inIter[j]) = test(N_CONTACTS, SOLVER, verb=VERB);
            print "Test %3d, time %3.2f, outIter %3d, inIter %3d" % (i, 1e3*time[j], outIter[j], inIter[j]);
            j += 1;
#            ret = cProfile.run("test()");
        except Exception as e:
            print e;
    print "\nMean computation time %.3f" % (1e3*np.mean(time));
    print "Mean outer iterations %d" % (np.mean(outIter));
    print "Mean inner iterations %d" % (np.mean(inIter));
    print "\nMax computation time %.3f" % (1e3*np.max(time));
    print "Max outer iterations %d" % (np.max(outIter));
    print "Max inner iterations %d" % (np.max(inIter));
