# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:20:09 2015

@author: adelpret
"""
import numpy as np
from sot_utils import hrp2_jointId_2_name, TAU_MAX

def enum(**enums):
    return type('Enum', (), enums)

ConstraintViolationType = enum(none=0, force=1, position=2, torque=3, velocity=4);

class ConstraintViolation(object):
    violationType = ConstraintViolationType.none;
    time = 0;
    info = '';
    
    def __init__(self, violationType, time, info):
        self.violationType = violationType;
        self.time = time;
        self.info = info;
        
    def toString(self):
        s = 'Time %.3f'%(self.time)+' ';
        if(self.violationType==ConstraintViolationType.none):
            s += "Unknown ";
        elif(self.violationType==ConstraintViolationType.force):
            s += "Force ";
        elif(self.violationType==ConstraintViolationType.position):
            s += "Position ";
        elif(self.violationType==ConstraintViolationType.torque):
            s += "Torque ";
        s += "constraint violation. "+self.info;
        return s;

class ForceConstraintViolation(ConstraintViolation):
    violationType = ConstraintViolationType.force;
    time = 0;
    contactName = '';
    w = 0;
    v = 0;
    w_d = None;
    H = None;
    H_d = None;
    
    def __init__(self, time, contactName, wrench, velocity, wrenchDes=None):
        self.time = time;
        self.contactName = contactName;
        self.w = wrench;
        self.v = velocity;
        self.w_d = wrenchDes;
        
    def toString(self):
        v = np.copy(self.v);
        w = np.copy(self.w);
        s = 'Time %.3f'%(self.time)+' FORCE VIOLATION ';
        if(self.w[2]!=0.0):
            w[:2]*=1e0/w[2]; 
            w[3:]*=1e0/w[2];        
        s += "%s v=(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) w=(%.2f,%.2f,%.1f,%.2f,%.2f,%.2f)" % (
            self.contactName, v[0],v[1],v[2],v[3],v[4],v[5],w[0],w[1],w[2],w[3],w[4],w[5]);
        if(self.w_d!=None):
            w = np.copy(self.w_d);
            if(w[2]!=0.0):
                w[:2]*=1e0/w[2];
                w[3:]*=1e0/w[2];
            s += " w_d=(%.2f,%.2f,%.1f,%.2f,%.2f,%.2f)" % (w[0],w[1],w[2],w[3],w[4],w[5]);
        return s;
        
class PositionConstraintViolation(ConstraintViolation):
    violationType = ConstraintViolationType.position;
    time = 0;
    jointId = 0;
    q = 0;
    dq = 0;
    dq_ctrl = None;
    ddq = 0;
    ddq_d = None;
    
    def __init__(self, time, jointId, q, dq, ddq, ddq_d=None):
        self.time = time;
        self.jointId = jointId;
        self.q = q;
        self.dq = dq;
        self.ddq = ddq;
        self.ddq_d = ddq_d;
        
    def toString(self):
        s = 'Time %.3f POS VIOLATION '%(self.time);
        s += "joint %d, dq=%.1f, ddq=%.1f " % (self.jointId, self.dq, self.ddq);
        if(self.ddq_d!=None):
            s += "ddq_d=%.1f "%self.ddq_d;
        if(self.dq_ctrl!=None):
            s += "dq_ctrl=%.1f "%self.dq_ctrl;
        return s;
        
class VelocityConstraintViolation(ConstraintViolation):
    violationType = ConstraintViolationType.velocity;
    time = 0;
    jointId = 0;
    q = 0;
    dq = 0;
    ddq = 0;
    dq_ctrl = None;
    ddq_d = None;
    
    def __init__(self, time, jointId, dq, ddq, ddq_d=None):
        self.time = time;
        self.jointId = jointId;
        self.dq = dq;
        self.ddq = ddq;
        self.ddq_d = ddq_d;
        
    def toString(self):
        s = 'Time %.3f VEL VIOLATION '%(self.time);
        s += "joint %d, dq=%.1f, ddq=%.1f " % (self.jointId, self.dq, self.ddq);
        if(self.ddq_d!=None):
            s += "ddq_d=%.1f "%self.ddq_d;
        if(self.dq_ctrl!=None):
            s += "dq_ctrl=%.1f "%self.dq_ctrl;
        return s;
        
class TorqueConstraintViolation(ConstraintViolation):
    violationType = ConstraintViolationType.torque;
    time = 0;
    jointId = 0;
    tau = 0;
    
    def __init__(self, time, jointId, tau):
        self.time = time;
        self.jointId = jointId;
        self.tau = tau;
        
    def toString(self):
        s = 'Time %.3f'%(self.time)+' TORQUE VIOLATION';
        s += "joint %d %s, tau=%.1f Nm, tauMax=%.1f Nm " % (self.jointId, hrp2_jointId_2_name(self.jointId), 
                                                            self.tau, TAU_MAX[self.jointId]);
        return s;