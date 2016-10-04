# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 22:18:04 2015

@author: adelpret
"""
import numpy as np
import matplotlib.pyplot as plt

class MinimumJerkTrajectoryGenerator(object):
    DISCRETE_TIME = False;
    
    x = []
    dx = [];
    ddx = [];
    
    x_prev = [];
    x_next = [];
    
    x_init = [];
    x_final = [];
    traj_time = 1;
    dt = 1e-3;
    t = 0;
        
    def __init__(self, dt, traj_time, x_init=None, x_final=None):
        self.dt = dt;
        self.traj_time = traj_time;
        if(x_init!=None):
            self.set_initial_point(x_init);
        if(x_final!=None):
            self.set_final_point(x_final);
        pass;
        
    def set_initial_point(self, x_init):
        self.x_init = np.copy(x_init);
        self.x      = np.copy(x_init);
        self.x_prev = np.copy(x_init);
        self.x_next = np.copy(x_init);
        self.t      = 0;
        
    def set_final_point(self, x_final):
        self.x_final = np.copy(x_final);
        
    def set_trajectory_time(self, traj_time):
        self.traj_time = traj_time;
        
    def get_next_point(self):
        if(self.t < self.traj_time):
            t = self.t;
            if(self.DISCRETE_TIME):
                t += self.dt;
            td  = t/self.traj_time;
            td2 = td**2;
            td3 = td2*td;
            td4 = td3*td;
            td5 = td4*td;
            p   = 10*td3 - 15*td4 + 6*td5;
            if(self.DISCRETE_TIME):
                self.x_prev = np.copy(self.x);
                self.x      = np.copy(self.x_next);
                self.x_next = self.x_init + (self.x_final-self.x_init)*p;
                self.dx     = (self.x_next-self.x)/self.dt;
                self.ddx    = (self.x_next-2*self.x+self.x_prev)/(self.dt**2);
            else:
                dp  = (30*td2 - 60*td3 + 30*td4)/self.traj_time;
                ddp = (60*td - 180*td2 + 120*td3)/self.traj_time**2;     
                self.x   = self.x_init + (self.x_final-self.x_init)*p;
                self.dx  = (self.x_final-self.x_init)*dp;
                self.ddx = (self.x_final-self.x_init)*ddp;
        else:
            self.x = self.x_final;
            self.dx = np.zeros(self.x.shape);
            self.ddx = np.zeros(self.x.shape);
        self.t += self.dt;
        return self.x;
        
if(0):
    dt = 1e-2;
    traj_time = 3.5;
    x_init = np.array([1.2]);
    x_final = np.array([3.8]);
    tj = MinimumJerkTrajectoryGenerator(dt, traj_time, x_init, x_final);
    
    N = int(traj_time/dt);
    x = np.zeros(N);
    dx = np.zeros(N);
    ddx = np.zeros(N);
    for i in range(N):
        x[i]   = tj.get_next_point();
        dx[i]  = tj.dx;
        ddx[i] = tj.ddx;
    
    f, ax = plt.subplots(3);
    ax[0].plot(x);
    ax[1].plot(dx);
    ax[2].plot(ddx);
    
    dx_fd  = np.diff(x,1,0)/dt;
    ddx_fd = np.diff(dx,1,0)/dt;
    ax[1].plot(dx_fd,'r');
    ax[2].plot(ddx_fd,'r');