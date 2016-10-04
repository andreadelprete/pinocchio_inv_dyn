# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:56:24 2016

Classes for computing numerical first and second derivates of a discrete-time trajectory.

@author: adelpret
"""

import numpy as np
from math import floor

''' Compute the first and second derivatives of the specified discrete-time trajectory x
    by finite differences.
    @param x A NxT numpy matrix, where N is the size of the signal and T is the number of time steps
    @param dt The duration of the time step in seconds
    @return (dx, ddx) Two NxT numpy matrices containing first and second derivatives of x.
'''
def computeFirstSecondDerivatives(x, dt):
    assert dt>0.0, "dt must be positive"
    assert len(x.shape)==2, "x must be a matrix"
    N   = x.shape[0];
    T   = x.shape[1];
    dx  = np.matlib.empty((N,T));
    ddx = np.matlib.empty((N,T));
    
    dx[:,:-1] = (x[:,1:] - x[:,:-1]) / dt;
    dx[:,-1]  = dx[:,-2];
    
    ddx[:,:-1] = (dx[:,1:] - dx[:,:-1]) / dt;
    ddx[:,-1]  = ddx[:,-2];
    
    return (dx, ddx);

''' Filter the specified discrete-time trajectory x by fitting a second-order polynomial to a sliding
    window of the specified length.
    @param x A NxT numpy matrix, where N is the size of the signal and T is the number of time steps
    @param dt The duration of the time step in seconds
    @param window_length An positive odd integer (>=3) representing the length of the sliding window
    @return (x, dx, ddx) A tuple of three NxT numpy matrices containing the filtered signal and its first two derivatives.
''' 
def computeSecondOrderPolynomialFitting(x, dt, window_length):
    assert dt>0.0, "dt must be positive"
    assert len(x.shape)==2, "x must be a matrix"
    assert window_length>2, "window length must be at least 3"
    assert window_length%2==1, "window length must be an odd number"
    N   = x.shape[0];
    T   = x.shape[1];
    w   = window_length;
    wh  = int(floor(0.5*(window_length-1)));    # half window length
    xp  = np.matlib.empty((N,T));
    dx  = np.matlib.empty((N,T));
    ddx = np.matlib.empty((N,T));
    
    for t in range(wh):
        A   = np.matlib.empty((t+wh+1,3));
        A[:,2] = 1.0;
        A[:,1] = dt * np.matrix(range(t+wh+1)).T;
        A[:,0] = np.square(A[:,1]);
        Ap = np.linalg.pinv(A);
        abc      = Ap * x[:,:t+wh+1].T;
        xp[:,t]  = (A[t,:] * abc).T;
        dx[:,t]  = (np.matrix([2*dt*t, 1]) * abc[:2,:]).T;
        ddx[:,t] = 2.0*abc[0,:].T;

    A   = np.matlib.empty((w,3));
    A[:,2] = 1.0;
    A[:,1] = dt * np.matrix(range(w)).T;
    A[:,0] = np.square(A[:,1]);
    Ap = np.linalg.pinv(A);
        
    for t in range(wh, T-wh):
        abc      = Ap * x[:,t-wh:t+wh+1].T;
        xp[:,t]  = (A[wh,:] * abc).T;
        dx[:,t]  = (np.matrix([2*dt*wh, 1]) * abc[:2,:]).T;
        ddx[:,t] = 2.0*abc[0,:].T;
        
    for t in range(T-wh, T):
        A   = np.matlib.empty((T-t+wh,3));
        A[:,2] = 1.0;
        A[:,1] = dt * np.matrix(range(T-t+wh)).T;
        A[:,0] = np.square(A[:,1]);
        Ap = np.linalg.pinv(A);
        abc      = Ap * x[:,t-wh:].T;
        xp[:,t]  = (A[wh,:] * abc).T;
        dx[:,t]  = (np.matrix([2*dt*wh, 1]) * abc[:2,:]).T;
        ddx[:,t] = 2.0*abc[0,:].T;
        
    return (xp, dx, ddx)