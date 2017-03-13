#
# Copyright (c) 2015-2016 CNRS
#
# This file is part of Pinocchio
# Pinocchio is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# Pinocchio is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# Pinocchio If not, see
# <http://www.gnu.org/licenses/>.

import time

from pinocchio.robot_wrapper import RobotWrapper as PinocchioRobotWrapper
import pinocchio.libpinocchio_pywrap as se3
import pinocchio.utils as utils
from pinocchio.explog import exp
import numpy as np


class RobotWrapper(PinocchioRobotWrapper):

    def mass(self, q, update_kinematics=True):
        if(update_kinematics):
            return se3.crba(self.model, self.data, q)
        return self.data.M

    def bias(self, q, v, update_kinematics=True):
        if(update_kinematics):
            return se3.nle(self.model, self.data, q, v)
        return self.data.nle
        
    def Jcom(self, q, update_kinematics=True):
        if(update_kinematics):
            return se3.jacobianCenterOfMass(self.model, self.data, q)
        return self.data.Jcom
        
    def momentumJacobian(self, q, v, update=True):
        if(update):
            se3.ccrba(self.model, self.data, q, v);
        return self.data.Ag;


    def computeAllTerms(self, q, v):
        return se3.computeAllTerms(self.model, self.data, q, v);
            
    ''' Compute the placements of all the operational frames and put the results in data.
        To be called after forwardKinematics. 
    '''
    def framesKinematics(self, q):
        se3.framesKinematics(self.model, self.data, q)
        
    def framePosition(self, index):
        f = self.model.frames[index]
        return self.data.oMi[f.parent].act(f.placement)

    def frameVelocity(self, index):
        f = self.model.frames[index]
        return f.placement.actInv(self.data.v[f.parent])
        
    ''' Return the spatial acceleration of the specified frame. '''
    def frameAcceleration(self, index):
        f = self.model.frames[index]
        return f.placement.actInv(self.data.a[f.parent])
        
    def frameClassicAcceleration(self, index):
        f = self.model.frames[index]
        a = f.placement.actInv(self.data.a[f.parent])
        v = f.placement.actInv(self.data.v[f.parent])
        a.linear += np.cross(v.angular.T, v.linear.T).T
        return a;
               
    ''' Call computeJacobians if update_geometry is true. If not, user should call computeJacobians first.
        Then call getJacobian and return the resulted jacobian matrix. Attention: if update_geometry is true, 
        the function computes all the jacobians of the model. It is therefore outrageously costly wrt a 
        dedicated call. Use only with update_geometry for prototyping.
    '''
    def frameJacobian(self, q, index, update_geometry=True, local_frame=True):
        return se3.frameJacobian(self.model, self.data, q, index, local_frame, update_geometry)
        
    # Create the scene displaying the robot meshes in gepetto-viewer
    def loadDisplayModel(self, nodeName, windowName="pinocchio"):
        # Open a window for displaying your model.
        try:
            # If the window already exists, do not do anything.
            self.windowID = self.viewer.gui.getWindowID(windowName)
            print "Warning: window '%s' already created. Cannot (re-)load the model." % windowName
        except:
             # Otherwise, create the empty window.
            self.windowID = self.viewer.gui.createWindow(windowName)

            # Start a new "scene" in this window, named "world", with just a floor.
            if "world" not in self.viewer.gui.getSceneList():
                self.viewer.gui.createSceneWithFloor("world")
            self.viewer.gui.addSceneToWindow("world", self.windowID)
    
        try:
            self.viewer.gui.createGroup(nodeName)
        except:
            pass

        # iterate over visuals and create the meshes in the viewer
        for visual in self.visual_model.geometryObjects :			
            try:
                meshName = self.viewerNodeNames(visual) 
                meshPath = visual.meshPath
                self.viewer.gui.addMesh(meshName, meshPath)
            except:
                pass

        # Finally, refresh the layout to obtain your first rendering.
        self.viewer.gui.refresh()
      
    def deactivateCollisionPairs(self, collision_pair_indexes):
        for i in collision_pair_indexes:
            self.collision_data.deactivateCollisionPair(i);
            
    def addAllCollisionPairs(self):
        self.collision_model.addAllCollisionPairs();
        self.collision_data = se3.GeometryData(self.collision_model);
        
    def isInCollision(self, q, stop_at_first_collision=True):
        return se3.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, stop_at_first_collision);

    def findFirstCollisionPair(self, consider_only_active_collision_pairs=True):
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(se3.computeCollision(self.collision_model, self.collision_data, i)):
                    return (i, self.collision_model.collisionPairs[i]);
        return None;
        
    def findAllCollisionPairs(self, consider_only_active_collision_pairs=True):
        res = [];
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(se3.computeCollision(self.collision_model, self.collision_data, i)):
                    res += [(i, self.collision_model.collisionPairs[i])];
        return res;

__all__ = ['RobotWrapper']
