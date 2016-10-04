# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 10:28:50 2015

Utility functions to compute the acceleration bounds for a system
with bounded position, velocity and accelerations. These computations
are based on the viability theory. A state is viable if, starting
from that state, it is possible to respect all the bounds (i.e. position, 
velocity and acceleration) in the future.

@author: adelpret
"""
import numpy as np
import acc_bounds_util

EPS = 1e-6;    # tolerance used to check violations

''' Return True if the state is viable, False otherwise. '''
def areStatesViable(q, dq, qMin, qMax, dqMax, ddqMax, verbose=False):
    dqMaxViab =   np.sqrt(np.maximum(0.0, 2.0*np.multiply(ddqMax, qMax-q)));
    dqMinViab = - np.sqrt(np.maximum(0.0, 2.0*np.multiply(ddqMax, q-qMin)));
    
    ind_q       = np.logical_or(q<qMin-EPS, q>qMax+EPS);
    ind_dq      = np.logical_or(dq>dqMax+EPS, dq<-dqMax-EPS);    
    ind_viab    = np.logical_or(dq>dqMaxViab+EPS, dq<dqMinViab+EPS);
    
    if(verbose):
        if(np.sum(ind_q)>0):
            print "WARNING: some states are not viable because they violate position bounds:", np.where(ind_q)[0], \
                "qMax-q", qMax[ind_q]-q[ind_q], "q-qMin", q[ind_q]-qMin[ind_q];
        elif(np.sum(ind_dq)>0):
            print "WARNING: some states are not viable because they violate velocity bounds:", np.where(ind_dq)[0], \
                 "dq", dq[ind_dq], "dqMax", dqMax[ind_dq];
        elif(np.sum(ind_viab)>0):
            print "WARNING: some states are not viable because they violate viability bounds:", np.where(ind_viab)[0], \
                "qMax-q", qMax[ind_viab]-q[ind_viab], "q-qMin", q[ind_viab]-qMin[ind_viab], "dq", dq[ind_viab], \
                "dqMaxViab", dqMaxViab[ind_viab], "dqMinViab", dqMinViab[ind_viab];

    return np.logical_or(ind_q, np.logical_or(ind_dq, ind_viab));    

''' Compute acceleration limits imposed by position bounds.
'''
def computeAccLimitsFromPosLimits(q, dq, qMin, qMax, ddqMax, dt, verbose=True):
    two_dt_sq   = 2.0/(dt**2);
    ddqMax_q3 = two_dt_sq*(qMax-q-dt*dq);
    ddqMin_q3 = two_dt_sq*(qMin-q-dt*dq);
    minus_dq_over_dt = -dq/dt;
    n = q.shape[0];
    ddqUB = np.matlib.zeros((n,1));
    ddqLB = np.matlib.zeros((n,1));
    
    ind  = (dq<=0.0).A.squeeze();
    ind2 = np.logical_and(ind, (ddqMin_q3<minus_dq_over_dt).A.squeeze());
    ind3 = np.logical_and(ind, np.logical_and(np.logical_not(ind2), (q!=qMin).A.squeeze()));
    ind4 = np.logical_and(ind, np.logical_and(np.logical_not(ind2), np.logical_not(ind3)));
    ddqUB[ind]  = ddqMax_q3[ind];
    ddqLB[ind2]  = ddqMin_q3[ind2];
    if(np.sum(ind3)>0):
        ddqMin_q2 = np.divide(np.square(dq[ind3]), 2.0*(q[ind3]-qMin[ind3]));
        ddqLB[ind3] = np.maximum(ddqMin_q2, minus_dq_over_dt[ind3]);
    ddqLB[ind4] = ddqMax[ind4];
    
    nind = np.logical_not(ind);
    ind2 = np.logical_and(nind, (ddqMax_q3>minus_dq_over_dt).A.squeeze());
    ind3 = np.logical_and(nind, np.logical_and(np.logical_not(ind2), (q!=qMax).A.squeeze()));
    ind4 = np.logical_and(nind, np.logical_and(np.logical_not(ind2), np.logical_not(ind3)));    
    ddqLB[nind] = ddqMin_q3[nind];
    ddqUB[ind2]  = ddqMax_q3[ind2];
    if(np.sum(ind3)>0):
        ddqMax_q2 = np.divide(-np.square(dq[ind3]), 2.0*(qMax[ind3]-q[ind3]));
        ddqUB[ind3] = np.minimum(ddqMax_q2, minus_dq_over_dt[ind3]);
    ddqUB[ind4] = -ddqMax[ind4];
    
#    # The following code is a slower (but more readable) version of the code above
#    for i in range(n):
#        if(dq[i]<=0.0):
#            ddqUB[i]  = ddqMax_q3[i];
#            if(ddqMin_q3[i] < minus_dq_over_dt[i]):
#                ddqLB[i]  = ddqMin_q3[i]
#            elif(q[i]!=qMin[i]):
#                ddqMin_q2 = dq[i]**2/(2*(q[i]-qMin[i]));
#                ddqLB[i]  = max(ddqMin_q2, minus_dq_over_dt[i]);
#            else:
#                # q=qMin -> you're gonna violate the position bound
#                ddqLB[i] = ddqMax[i];
#        else:
#            ddqLB[i]  = ddqMin_q3[i];
#            if(ddqMax_q3[i] > minus_dq_over_dt[i]):
#                ddqUB[i]  = ddqMax_q3[i];
#            elif(q[i]!=qMax[i]):
#                ddqMax_q2 = -dq[i]**2/(2*(qMax[i]-q[i]));
#                ddqUB[i]  = min(ddqMax_q2, minus_dq_over_dt[i]);
#            else:
#                # q=qMax -> you're gonna violate the position bound
#                ddqUB[i] = -ddqMax[i];
            
    return (ddqLB, ddqUB);
    
    
''' Acceleration limits imposed by viability.
    ddqMax is the maximum acceleration that will be necessary to stop the joint before
    hitting the position limits.
    
     -sqrt( 2*ddqMax*(q-qMin) ) < dq[t+1] < sqrt( 2*ddqMax*(qMax-q) )
    ddqMin[2] = (-sqrt(max(0.0, 2*MAX_ACC*(q[i]+DT*dq[i]-qMin))) - dq[i])/DT;
    ddqMax[2] = (sqrt(max(0.0, 2*MAX_ACC*(qMax-q[i]-DT*dq[i]))) - dq[i])/DT;    
'''
def computeAccLimitsFromViability(q, dq, qMin, qMax, ddqMax, dt, verbose=True):
    dt_square = dt**2;
    dt_dq = dt*dq;
    minus_dq_over_dt = -dq/dt;
    dt_two_dq = 2*dt_dq;
    two_ddqMax = 2*ddqMax;
    dt_ddqMax_dt = ddqMax*dt_square;
    dq_square = np.square(dq);
    q_plus_dt_dq = q + dt_dq;
    
    two_a = 2*dt_square;
    b = dt_two_dq + dt_ddqMax_dt;
    c = dq_square - np.multiply(two_ddqMax, qMax - q_plus_dt_dq);
    delta = np.square(b) - 2*two_a*c;
    ind = (delta>=0.0).A.squeeze();
    nind = np.logical_not(ind);
    n = q.shape[0];
    ddq_1 = np.matlib.zeros((n,1));
    ddq_1[ind] = (-b[ind] + np.sqrt(delta[ind])) / two_a;
    ddq_1[nind] = minus_dq_over_dt[nind];
    if(np.sum(nind)>0 and verbose):
        print "Error: state(s) not viable because delta is negative", np.where(nind)[0];
    
    b = dt_two_dq - dt_ddqMax_dt;
    c = dq_square - np.multiply(two_ddqMax, q_plus_dt_dq - qMin);
    delta = np.square(b) - 2*two_a*c;
    ind = (delta>=0.0).A.squeeze();
    nind = np.logical_not(ind);
    ddq_2 = np.matlib.zeros((n,1));
    ddq_2[ind] = (-b[ind] - np.sqrt(delta[ind])) / two_a;
    ddq_2[nind] = minus_dq_over_dt[nind];
    if(np.sum(nind)>0 and verbose):
        print "Error: state(s) not viable because delta is negative", np.where(nind)[0];
        
    ddqUB = np.maximum(ddq_1, minus_dq_over_dt);
    ddqLB = np.minimum(ddq_2, minus_dq_over_dt);
    return (ddqLB, ddqUB);
    
        
''' Given the current position and velocity, the bounds of position,
    velocity and acceleration and the control time step, compute the
    bounds of the acceleration such that all the bounds are respected
    at the next time step and can be respected in the future.
    ddqStop is the maximum acceleration that will be necessary to stop the joint before
    hitting the position limits, whereas ddqMax is the absolute maximum acceleration.
'''
def computeAccLimits(q, dq, qMin, qMax, dqMax, ddqMax, dt, verbose=True, ddqStop=None, IMPOSE_POSITION_BOUNDS=True,
                     IMPOSE_VELOCITY_BOUNDS=True, IMPOSE_VIABILITY_BOUNDS=True, IMPOSE_ACCELERATION_BOUNDS=True):    
    if(verbose):
        viabViol = areStatesViable(q, dq, qMin, qMax, dqMax, ddqMax, verbose);
#    if(np.sum(viabViol)>0 and verbose):
#        print "WARNING: some states are not viable:", np.where(viabViol)[0];
        
    if(ddqStop==None):
        ddqStop=ddqMax;
        
    n = q.shape[0];
    ddqUB = np.matlib.zeros((n,4)) + 1e100;
    ddqLB = np.matlib.zeros((n,4)) - 1e100;
    
    # Acceleration limits imposed by position bounds
    if(IMPOSE_POSITION_BOUNDS):
        (ddqLB[:,0], ddqUB[:,0]) = computeAccLimitsFromPosLimits(q, dq, qMin, qMax, ddqMax, dt, verbose);
    
    # Acceleration limits imposed by velocity bounds
    # dq[t+1] = dq + dt*ddq < dqMax
    # ddqMax = (dqMax-dq)/dt
    # ddqMin = (dqMin-dq)/dt = (-dqMax-dq)/dt
    if(IMPOSE_VELOCITY_BOUNDS):
        ddqLB[:,1] = (-dqMax-dq)/dt;
        ddqUB[:,1] = (dqMax-dq)/dt;
    
    # Acceleration limits imposed by viability
    if(IMPOSE_VIABILITY_BOUNDS):
        (ddqLB[:,2], ddqUB[:,2]) = computeAccLimitsFromViability(q, dq, qMin, qMax, ddqStop, dt, verbose);
     
    # Acceleration limits
    if(IMPOSE_ACCELERATION_BOUNDS):
        ddqLB[:,3] = -ddqMax;
        ddqUB[:,3] = ddqMax;
    
    # Take the most conservative limit for each joint
    ddqLBFinal = np.max(ddqLB, 1);
    ddqUBFinal = np.min(ddqUB, 1);
    
    # In case of conflict give priority to position bounds
    ind = (ddqUBFinal<ddqLBFinal).A.squeeze()
    if(np.sum(ind)>0):        
        if(verbose):
            print "Conflict between pos/vel/acc bounds (ddqMin, ddqMax)=", (np.where(ind)[0], ddqLBFinal[ind],ddqUBFinal[ind]);
        
        # eliminate acceleration limits and recompute final bounds
        ddqUB[ind,3] = 1e100;
        ddqLB[ind,3] = -1e100;
        ddqLBFinal[ind] = np.max(ddqLB[ind], 1);
        ddqUBFinal[ind] = np.min(ddqUB[ind], 1);
        ind2 = (ddqUBFinal<ddqLBFinal).A.squeeze()
        if(np.sum(ind2)>0):
            if(verbose):
                print "Conflict persists after eliminating acc limits (ddqMin, ddqMax)=", (np.where(ind2)[0], ddqLBFinal[ind2],ddqUBFinal[ind2]);
            
            # eliminate viability limits and recompute final bounds
            ddqUB[ind2,2] = 1e100;
            ddqLB[ind2,2] = -1e100;
            ddqLBFinal[ind2] = np.max(ddqLB[ind2], 1);
            ddqUBFinal[ind2] = np.min(ddqUB[ind2], 1);
            ind3 = (ddqUBFinal<ddqLBFinal).A.squeeze()
            if(np.sum(ind3)>0):
                if(verbose):
                    print "Conflict persists after eliminating viab limits (ddqMin, ddqMax)=", (np.where(ind3)[0], ddqLBFinal[ind3],ddqUBFinal[ind3]);
                # use position limits 
                ddqLBFinal[ind3] = ddqLB[ind3,0];
                ddqUBFinal[ind3] = ddqUB[ind3,0];

        if(verbose):
            print "                     New bounds are (ddqMin, ddqMax)=", (ddqLBFinal[ind],ddqUBFinal[ind]);

#    ddqLBs = np.matlib.zeros((n,1));
#    ddqUBs = np.matlib.zeros((n,1));
#    for i in range(n):
#        (ddqLBs[i], ddqUBs[i]) = acc_bounds_util.computeAccLimits(q[i], dq[i], qMin[i], qMax[i], dqMax[i], ddqMax[i], 
#                                                                  dt, verbose, ddqStop[i], IMPOSE_POSITION_BOUNDS, 
#                                                                  IMPOSE_VELOCITY_BOUNDS, IMPOSE_VIABILITY_BOUNDS, 
#                                                                  IMPOSE_ACCELERATION_BOUNDS);
#        if(abs(ddqLBs[i]-ddqLBFinal[i])>EPS or abs(ddqUBs[i]-ddqUBFinal[i])>EPS):
#            print "Error computing acceleration lower bound for joint", i, ddqLBs[i], ddqLBFinal[i];
#            print "Error computing acceleration upper bound for joint", i, ddqUBs[i], ddqUBFinal[i];
#            (ddqLBs[i], ddqUBs[i]) = acc_bounds_util.computeAccLimits(q[i], dq[i], qMin[i], qMax[i], dqMax[i], ddqMax[i], 
#                                                                  dt, True, ddqStop[i], IMPOSE_POSITION_BOUNDS, 
#                                                                  IMPOSE_VELOCITY_BOUNDS, IMPOSE_VIABILITY_BOUNDS, 
#                                                                  IMPOSE_ACCELERATION_BOUNDS);
#            computeAccLimits(q, dq, qMin, qMax, dqMax, ddqMax, dt, True, ddqStop, IMPOSE_POSITION_BOUNDS,
#                             IMPOSE_VELOCITY_BOUNDS, IMPOSE_VIABILITY_BOUNDS, IMPOSE_ACCELERATION_BOUNDS)
    
    return (ddqLBFinal, ddqUBFinal);
    