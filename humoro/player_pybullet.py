import pybullet_utils.bullet_client as bc
import pybullet
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
        self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        #p.connect(p.GUI)
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
        self.gaze_objs = []
        self.trans_world = np.array([0., 0., 0.])  # this can be used to translate the world

        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
        self.plane = self.p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/plane.urdf", basePosition=[0, 0, 0]+self.trans_world)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)


    def spawnObject(self, name, meshfile=None, color=[.8, .8, .8, 1.]):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
        if meshfile is None:
            visualShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_SPHERE, rgbaColor=color, specularColor=[1, 1, 1], radius=0.02)
        else:
            visualShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_MESH, fileName=meshfile, rgbaColor=color, specularColor=[1, 1, 1])
        cuid = -1  # self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents = [1, 1, 1])
        bodyid = self.p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cuid, baseInertialFramePosition=[0, 0, 0], baseVisualShapeIndex=visualShapeId, basePosition=[0, 1, 0], useMaximalCoordinates=True)
        self._objects[name] = bodyid
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)
        
    def spawnHuman(self, name="human1", urdf=None, upper_body=False, color=None, hide=True):
        """ spawns a human

        Keyword arguments:
        name -- name of the spawned human (default "human1")
        urdf -- path to urdf file. if None, use urdfs from data folder(default None)
        upper_body -- if true, load upper body model else load full body. Only used when urdf=None (default False)
        color -- color of the human in rgba (default None)
        hide -- hide the human if no trajectory for frame is assigned (default True)
        """
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)  # disable rendering during loading makes it much faster
        if urdf is None:
            zpos = 0
            if hide:
                zpos = -10.  # Todo: cleaner way to hide humans?
            if upper_body is True:
                self._humans[name] = self.p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human_upper_body.urdf", basePosition=[0, 0, zpos])
            else:
                self._humans[name] = self.p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human.urdf", basePosition=[0, 0, zpos])
        else:
            self._humans[name] = self.p.loadURDF(urdf, basePosition=[0, 0, -10])

        self.inv_index = create_inv_index(self._humans[name], self.p)
        if color is not None:
            self.change_color(self._humans[name], color)
        self._hidehumans[name] = hide
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)
        return self._humans[name]

    def change_color(self, human, color):
        """ changes the color of a human

        Keyword arguments:
        human -- name of the human
        color -- new color of human in rgba
        """
        for j in range(self.p.getNumJoints(human)):
            self.p.changeVisualShape(human, j, rgbaColor=color)

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

    def addPlaybackTrajGaze(self, traj):
        """ adds an trajectory to the player for playback of gaze

        Keyword arguments:
        traj -- playback trajectory
        """
        self.gaze_trajs.append(traj)
        gazeshape = self.p.createVisualShape(shapeType=self.p.GEOM_CYLINDER, rgbaColor=[0., 0., 1., 1.], specularColor=[1, 1, 1], radius=0.01, length=10.)
        self.gaze_objs.append(self.p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0], baseVisualShapeIndex=gazeshape, basePosition=[0, 0, 0], useMaximalCoordinates=True))

            
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
        while (self.p.isConnected()):
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
        #self.p.stepSimulation()

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
                        
            self.p.resetJointStatesMultiDof(self._humans[human], idxes, values)
        # hide not visible humans
        for human in self._humans:
            if not self._hidehumans[human]:
                continue
            if human in visibleHumans:
                self.p.resetBasePositionAndOrientation(self._humans[human], self.trans_world, [0, 0, 0, 1])
            else:
                self.p.resetBasePositionAndOrientation(self._humans[human], [0, 0, -10], [0, 0, 0, 1])  # TODO: is there a cleaner way to hide humans?

        # playback object trajectories
        for obj, traj in self._playbackTrajsObj:
            if frame >= traj.startframe and frame < traj.endframe:
                trajframe = int(frame - traj.startframe)
                if obj in self._objects:
                    self.p.resetBasePositionAndOrientation(self._objects[obj], traj.data[trajframe][0:3]+self.trans_world, traj.data[trajframe][3:7])
                if obj == "goggles":  # playback gaze trajectory
                    for gi in range(len(self.gaze_trajs)):
                        gaze_traj = self.gaze_trajs[gi]
                        if frame >= gaze_traj.startframe and frame < gaze_traj.endframe:
                            trajframegaze = int(frame - gaze_traj.startframe)
                            if gaze_traj.data[trajframegaze, -1] > 0.:
                                rotmat = mu.quaternion_matrix(gaze_traj.data_fixed['calibration'])
                                rotmat = np.dot(mu.quaternion_matrix(traj.data[trajframe, 3:7]), rotmat)
                                endpos = gaze_traj.data[trajframegaze]
                                if endpos[2] < 0:
                                    endpos *= -1  # mirror gaze point if wrong direction
                                endpos = np.dot(rotmat, endpos)  # gaze calibration
                                direction = traj.data[trajframe][0:3]+self.trans_world-endpos
                                direction /= np.linalg.norm(direction)
                                self.p.resetBasePositionAndOrientation(self.gaze_objs[gi], traj.data[trajframe][0:3]+self.trans_world - direction*5., mu.points2quat([0.,0.,1.], direction))
