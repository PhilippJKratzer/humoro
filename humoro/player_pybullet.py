import pybullet as p
import os
import time
import numpy as np
import math
from humoro.pybullet_helpers import create_inv_index
import humoro._math_utils as mu

class Player:
    def __init__(self, fps=120):
        """ init function

        Keyword arguments:
        fps -- frames per second of the trajectories (default 120)
        """
        p.connect(p.GUI)
        #p.setGravity(0,0,-100)

        self._humans = {}
        self._hidehumans = {}
        self._objects = {}
        self._fps = fps
        self._playbackTrajs = []
        self._playbackTrajs_interp = []
        self._playbackTrajsObj = []
        self._start_playback = None
        self._end_playback = None
        self.gaze_trajs = []
        self.trans_world = np.array([0., 0., 0.])  # this can be used to translate the world

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.plane = p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/plane.urdf", basePosition=[0, 0, 0]+self.trans_world)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


    def spawnObject(self, name, meshfile=None, color=[.8, .8, .8, 1.]):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        if meshfile is None:
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=color, specularColor=[1, 1, 1], radius=0.02)
        else:
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=meshfile, rgbaColor=color, specularColor=[1, 1, 1])
        cuid = -1  # p.createCollisionShape(p.GEOM_BOX, halfExtents = [1, 1, 1])
        bodyid = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cuid, baseInertialFramePosition=[0, 0, 0], baseVisualShapeIndex=visualShapeId, basePosition=[0, 1, 0], useMaximalCoordinates=True)
        self._objects[name] = bodyid
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
    def spawnHuman(self, name="human1", urdf=None, upper_body=False, color=None, hide=True):
        """ spawns a human

        Keyword arguments:
        name -- name of the spawned human (default "human1")
        urdf -- path to urdf file. if None, use urdfs from data folder(default None)
        upper_body -- if true, load upper body model else load full body. Only used when urdf=None (default False)
        color -- color of the human in rgba (default None)
        hide -- hide the human if no trajectory for frame is assigned (default True)
        """
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # disable rendering during loading makes it much faster
        if urdf is None:
            zpos = 0
            if hide:
                zpos = -10.  # Todo: cleaner way to hide humans?
            if upper_body is True:
                self._humans[name] = p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human_upper_body.urdf", basePosition=[0, 0, zpos])
            else:
                self._humans[name] = p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human.urdf", basePosition=[0, 0, zpos])
        else:
            self._humans[name] = p.loadURDF(urdf, basePosition=[0, 0, -10])

        self.inv_index = create_inv_index(self._humans[name], p)
        if color is not None:
            self.change_color(self._humans[name], color)
        self._hidehumans[name] = hide
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return self._humans[name]

    def change_color(self, human, color):
        """ changes the color of a human

        Keyword arguments:
        human -- name of the human
        color -- new color of human in rgba
        """
        for j in range(p.getNumJoints(human)):
            p.changeVisualShape(human, j, rgbaColor=color)

    def addPlaybackTraj(self, traj, human="human1"):
        """ adds a human trajectory to the player for playback

        Keyword arguments:
        traj -- playback trajectory
        human -- name of human to add trajectory to (default human1)
        """
        self._playbackTrajs.append((human, traj))
        if self._start_playback is None or self._start_playback > traj.startframe:
            self._start_playback = traj.startframe
        if self._end_playback is None or self._end_playback < traj.endframe:
            self._end_playback = traj.endframe

    def addPlaybackTrajObj(self, traj, obj="object1"):
        """ adds an object trajectory to the player for playback

        Keyword arguments:
        traj -- playback trajectory
        obj -- name of the object to add trajectory to (default object1)
        """
        self._playbackTrajsObj.append((obj, traj))
        if self._start_playback is None or self._start_playback > traj.startframe:
            self._start_playback = traj.startframe
        if self._end_playback is None or self._end_playback < traj.endframe:
            self._end_playback = traj.endframe
            
    def removePlaybackTrajs(self):
        self._playbackTrajs = []
        self._playbackTrajsObj = []
        self._start_playback = None
        self._end_playback = None
            

        
    def addPlaybackTrajObjList(self, trajs, objs=[]):
        """ adds a list of trajectories to a list of objects

        Keyword arguments:
        trajs -- list of playback trajectories
        objs -- list of object names
        """
        for traj, obj in zip(trajs, objs):
            self._playbackTrajsObj.append((obj, traj))
            if self._start_playback is None or self._start_playback > traj.startframe:
                self._start_playback = traj.startframe
            if self._end_playback is None or self._end_playback < traj.endframe:
                self._end_playback = traj.endframe

    def play(self, duration=-1, startframe=-1):
        """ start playback of trajectories """
        start_time = time.time()
        if startframe == -1:
            startframe = self._start_playback
        while (p.isConnected()):
            frameReal = (time.time() - start_time) * self._fps
            frame =  startframe + int(frameReal)
            if duration != -1 and frame > duration + startframe:
                break
            elif frame >= self._end_playback:
                break
            else:
                self.showFrame(frame)

    def play_controls(self, path_segments=None):
        import humoro.play_controls
        humoro.play_controls.startwindow(path_segments=path_segments, playback_func=self.showFrame, time_start=self._start_playback, time_end=self._end_playback, fps=self._fps)

    def showFrame(self, frame):
        visibleHumans = []
        #p.stepSimulation()

        # playback human trajectories
        for human, traj in self._playbackTrajs:
            idxes = []
            values = []

            if frame >= traj.startframe and frame < traj.endframe:
                visibleHumans.append(human)
                trajinterp, trajframe = math.modf(frame - traj.startframe)
                trajframe = int(trajframe)
                for key in traj.data_fixed:
                    idx = self.inv_index.get(key, -1)
                    if idx != -1:
                        idxes.append(idx)
                        values.append([traj.data_fixed[key]])

                for i in range(traj.data.shape[1]):
                    idx = self.inv_index.get(traj.description[i], -1)
                    if idx != -1:
                        idxes.append(idx)
                        values.append([traj.data[trajframe, i]])
                        
            p.resetJointStatesMultiDof(self._humans[human], idxes, values)
        # hide not visible humans
        for human in self._humans:
            if not self._hidehumans[human]:
                continue
            if human in visibleHumans:
                p.resetBasePositionAndOrientation(self._humans[human], self.trans_world, [0, 0, 0, 1])
            else:
                p.resetBasePositionAndOrientation(self._humans[human], [0, 0, -10], [0, 0, 0, 1])  # TODO: is there a cleaner way to hide humans?

        # playback object trajectories
        for obj, traj in self._playbackTrajsObj:
            if frame >= traj.startframe and frame < traj.endframe:
                trajframe = int(frame - traj.startframe)
                if obj in self._objects:
                    p.resetBasePositionAndOrientation(self._objects[obj], traj.data[trajframe][0:3]+self.trans_world, traj.data[trajframe][3:7])
                if obj == "goggles":  # playback gaze trajectory
                    for gaze_traj in self.gaze_trajs:
                        if frame >= gaze_traj.startframe and frame < gaze_traj.endframe:
                            trajframegaze = int(frame - gaze_traj.startframe)
                            if gaze_traj.data[trajframegaze, -1] > 0.:
                                rotmat = mu.quaternion_matrix(gaze_traj.data_fixed['calibration'])
                                rotmat = np.dot(mu.quaternion_matrix(traj.data[trajframe, 3:7]), rotmat)
                                endpos = gaze_traj.data[trajframegaze]
                                if endpos[2] < 0:
                                    endpos *= -1  # mirror gaze point if wrong direction
                                endpos = np.dot(rotmat, endpos)  # gaze calibration

                                p.addUserDebugLine(traj.data[trajframe][0:3]+self.trans_world, endpos+self.trans_world, lifeTime=0.05, lineWidth=4, lineColorRGB=[0., 0., 0.5])  # TODO: this uses a Debug Line for displaying the gaze ray, is there a better option?

        
