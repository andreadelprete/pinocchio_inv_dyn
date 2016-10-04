# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:34:46 2015

@author: adelpret
"""

import pinocchio as se3
from pinocchio import RobotWrapper
from time import sleep
from time import time
import os
import numpy as np
from numpy.matlib import zeros
from first_order_low_pass_filter import FirstOrderLowPassFilter

ENABLE_VIEWER = True;
    
def xyzRpyToViewerConfig(xyz, rpy):
    R = se3.utils.rpyToMatrix(rpy);
    H = se3.SE3(R, xyz.reshape(3,1));
    pinocchioConf = se3.utils.se3ToXYZQUAT(H);
    return se3.utils.XYZQUATToViewerConfiguration(pinocchioConf);
    
class Viewer(object):
    
    viewer = None;
    robots = None;
    robot = None;

    name = '';
    PLAYER_FRAME_RATE = 20;
    
    COM_SPHERE_RADIUS = 0.01;
    COM_SPHERE_COLOR = (1, 0, 0, 1);

    CAMERA_LOW_PASS_FILTER_CUT_FREQUENCY = 10.0;
    CAMERA_FOLLOW_ROBOT = 'robot1';   # name of the robot to follow with the camera
    CAMERA_LOOK_AT_HEIGHT = 0.5;           # height the camera should look at when following a robot
    CAMERA_REL_POS = [4, 2, 1.75];  # distance from robot to camera
    CAMERA_LOOK_AT_OFFSET = [0, 0]; # offset to robot xy position looked by camera    
    
    objectsAttachedToJoints = [];

    def __init__(self, name, robotWrapper, robotName='robot1'):
        self.name = name;
        self.filter = FirstOrderLowPassFilter(0.002, self.CAMERA_LOW_PASS_FILTER_CUT_FREQUENCY, np.zeros(2));
        if(ENABLE_VIEWER):
            self.robot = robotWrapper;
            self.robot.initDisplay("world/"+robotName, loadModel=False);
            self.robot.loadDisplayModel("world/"+robotName, robotName);
            self.robot.viewer.gui.setLightingMode('world/floor', 'OFF');
            self.robots = {robotName: self.robot};                
                
    def addRobot(self, robotName, urdfModelPath, modelPath):
        if(ENABLE_VIEWER):
            newRobot = RobotWrapper(urdfModelPath, modelPath);
            newRobot.initDisplay("world/"+robotName, loadModel=False);
            newRobot.viewer.gui.addURDF("world/"+robotName, urdfModelPath, modelPath);
            self.robots[robotName] = newRobot;
                        
    def updateRobotConfig(self, q, robotName='robot1', refresh=True):
        if(ENABLE_VIEWER):
            self.robots[robotName].display(q);
            
            for o in self.objectsAttachedToJoints:
                pinocchioConf = se3.utils.se3ToXYZQUAT(self.robots[o['robot']].data.oMi[o['joint']]*o['H']);
                viewerConf = se3.utils.XYZQUATToViewerConfiguration(pinocchioConf);
                self.updateObjectConfig(o['object'], viewerConf, False);
            if(refresh):
                self.robot.viewer.gui.refresh();
                
            if(self.CAMERA_FOLLOW_ROBOT==robotName):
                xy = self.filter.filter_data(q[:2]);
                xy[0] += self.CAMERA_LOOK_AT_OFFSET[0];
                xy[1] += self.CAMERA_LOOK_AT_OFFSET[1];
                # pos: position of the object to look at
                # eye: camera position
                self.robot.viewer.gui.moveCamera(self.robot.windowID, xy[0], xy[1], self.CAMERA_LOOK_AT_HEIGHT, \
                                self.CAMERA_REL_POS[0]+xy[0], self.CAMERA_REL_POS[1]+xy[1], self.CAMERA_REL_POS[2], \
                                0, 0, 1);
    
    def moveCamera(self, target, cameraPos, upwardDirection=(0,0,1)):
        if(ENABLE_VIEWER):
            self.robot.viewer.gui.moveCamera(self.robot.windowID, target[0], target[1], target[2], \
                                             cameraPos[0], cameraPos[1], cameraPos[2], \
                                             upwardDirection[0], upwardDirection[1], upwardDirection[2]);
                                
    def play(self, q, dt, slow_down_factor=1, print_time_every=-1.0, robotName='robot1'):
        self.addSphere('com', self.COM_SPHERE_RADIUS, zeros((3,1)), zeros((3,1)), self.COM_SPHERE_COLOR, 'OFF');
        if(ENABLE_VIEWER):
            trajRate = 1.0/dt
            rate = int(slow_down_factor*trajRate/self.PLAYER_FRAME_RATE);
            lastRefreshTime = time();
            timePeriod = 1.0/self.PLAYER_FRAME_RATE;
            for t in range(0,q.shape[1],rate):                
                self.updateRobotConfig(q[:,t], robotName, refresh=False);
                com = self.robots[robotName].com(q[:,t]);
                self.updateObjectConfig('com', (com[0,0], com[1,0], 0, 0,0,0,1));
                timeLeft = timePeriod - (time()-lastRefreshTime);
                if(timeLeft>0.0):
                    sleep(timeLeft);
                self.robot.viewer.gui.refresh();
                lastRefreshTime = time();
                if(print_time_every>0.0 and t*dt%print_time_every==0.0):
                    print "%.1f"%(t*dt);
                    
    ''' multi_q is a dictionary that maps the name of the robot to its trajectory '''
    def playMultiRobot(self, multi_q, dt, slow_down_factor=1, print_time_every=-1.0):
        if(ENABLE_VIEWER):
            trajRate = 1.0/dt
            rate = int(slow_down_factor*trajRate/self.PLAYER_FRAME_RATE);
            lastRefreshTime = time();
            timePeriod = 1.0/self.PLAYER_FRAME_RATE;
            for t in range(0,multi_q.items()[0][1].shape[0],rate):
                for robotName, q in multi_q.items():                
                    self.updateRobotConfig(q[t,:], robotName, refresh=False);
                timeLeft = timePeriod - (time()-lastRefreshTime);
                if(timeLeft>0.0):
                    sleep(timeLeft);
#                else:
#                    print "Viewer::playMultiRobot too slow, timeLeft=",timeLeft;
                self.robot.viewer.gui.refresh();
                lastRefreshTime = time();
                if(print_time_every>0.0 and t*dt%print_time_every==0.0):
                    print "%.1f"%(t*dt);
                
    def activateMouseCamera(self):
        if(ENABLE_VIEWER):
            self.robot.viewer.gui.activateMouseCamera(self.robot.windowID);       
            
    def setLightingMode(self, name, mode='OFF'):
        if(ENABLE_VIEWER):
            self.robot.viewer.gui.setLightingMode('world/'+name, mode);

    ''' Add a box in the simulation environment
        @param name The name of the box
        @param size The 3d size of the box
        @param xyz A 3d numpy array containing the 3d position 
        @param rpy A 3d numpy array containing the orientation as roll-pitch-yaw angles
        @param color A 4d tuple containing the  RGB-alpha color
        @param lightingMode Either 'ON' or 'OFF', enable/disable the effect of light on the box
    '''
    def addBox(self, name, size, xyz, rpy, color=(0,0,0,1.0), lightingMode='ON'):
        if(ENABLE_VIEWER):
            position = xyzRpyToViewerConfig(xyz, rpy);
            self.robot.viewer.gui.addBox('world/'+name, size[0]/2, size[1]/2, size[2]/2, color);
            self.robot.viewer.gui.applyConfiguration('world/'+name, position);
            self.robot.viewer.gui.setLightingMode('world/'+name, lightingMode);
            self.robot.viewer.gui.refresh();
        
    def addSphere(self,name, radius, xyz, rpy, color=(0,0,0,1.0), lightingMode='ON'):
        if(ENABLE_VIEWER):
            position = xyzRpyToViewerConfig(xyz, rpy);
            self.robot.viewer.gui.addSphere('world/'+name, radius, color);
            self.robot.viewer.gui.applyConfiguration('world/'+name, position)
            self.robot.viewer.gui.setLightingMode('world/'+name, lightingMode);
            self.robot.viewer.gui.refresh();
        
    def addCylinder(self,name, radius, length, xyz, rpy, color=(0,0,0,1.0), lightingMode='ON'):
        if(ENABLE_VIEWER):
            position = xyzRpyToViewerConfig(xyz, rpy);
            self.robot.viewer.gui.addCylinder('world/'+name, radius, length, color);
            self.robot.viewer.gui.applyConfiguration('world/'+name, position)
            self.robot.viewer.gui.setLightingMode('world/'+name, lightingMode);
            self.robot.viewer.gui.refresh();
        
    def addMesh(self, name, filename, xyz=np.zeros(3), rpy=np.zeros(3)):
        if(ENABLE_VIEWER):
            position = xyzRpyToViewerConfig(xyz, rpy);
            self.robot.viewer.gui.addMesh("world/"+name, filename);
            self.robot.viewer.gui.applyConfiguration('world/'+name, position);
            self.robot.viewer.gui.refresh();

    def updateObjectConfig(self, name, config, refresh=True):
        if(ENABLE_VIEWER):
            self.robot.viewer.gui.applyConfiguration('world/'+name, config);
            if(refresh):
                self.robot.viewer.gui.refresh();
        
    def updateObjectConfigRpy(self, name, xyz=np.zeros(3), rpy=np.zeros(3)):
        if(ENABLE_VIEWER):
            config = xyzRpyToViewerConfig(xyz, rpy);
            self.updateObjectConfig(name, config);
        
    def attachObjectToJoint(self, objectName, jointName, xyz, rpy, robotName='robot1'):
        if(ENABLE_VIEWER):
            R = se3.utils.rpyToMatrix(rpy);
            H = se3.SE3(R, xyz.reshape(3,1));
            self.objectsAttachedToJoints += [{'object': objectName, 
                                              'joint': self.robots[robotName].index(jointName),
                                              'H': H,
                                              'robot': robotName}];
    #        pinocchioConf = se3.utils.se3ToXYZQUAT(self.robot.data.oMi[self.robot.index(jointName)]);
    #        viewerConf = se3.utils.XYZQUATToViewerConfiguration(pinocchioConf);
            self.updateObjectConfigRpy(objectName, xyz, rpy);
        
    def startCapture(self, filename, extension='jpeg', path='/home/adelpret/capture/'):
        if(ENABLE_VIEWER):
            if(not os.path.exists(path)):
                os.makedirs(path);
            self.robot.viewer.gui.startCapture(self.robot.windowID, path+filename, extension);
        
    def stopCapture(self):
        if(ENABLE_VIEWER):
            self.robot.viewer.gui.stopCapture(self.robot.windowID);
