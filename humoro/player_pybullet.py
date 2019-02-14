import pybullet as p
import os
import time


class Player:
    def __init__(self, fps=120):
        """ init function

        Keyword arguments:
        fps -- frames per second of the trajectories (default 120)
        """
        p.connect(p.GUI)

        self._humans = {}
        self._fps = fps
        self._playbackTrajs = []
        self._start_playback = None
        self._end_playback = None
        self.plane = p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/plane.urdf")

    def spawnHuman(self, name="human1", urdf=None, upper_body=False, color=None):
        """ spawns a human

        Keyword arguments:
        name -- name of the spawned human (default "human1")
        urdf -- path to urdf file. if None, use urdfs from data folder(default None)
        upper_body -- if true, load upper body model else load full body. Only used when urdf=None (default False)
        color -- color of the human in rgba (default None)
        """
        if urdf is None:
            if upper_body is True:
                self._humans[name] = p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human_upper_body.urdf", basePosition=[0, 0, -10])
            else:
                self._humans[name] = p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human.urdf", basePosition=[0, 0, -10])
        else:
            self._humans[name] = p.loadURDF(urdf, basePosition=[0, 0, -10])

        self._inv_index = self._create_inv_index(self._humans[name])
        if color is not None:
            self.change_color(self._humans[name], color)

    def change_color(self, human, color):
        """ changes the color of a human

        Keyword arguments:
        human -- name of the human
        color -- new color of human in rgba
        """
        for j in range(p.getNumJoints(human)):
            p.changeVisualShape(human, j, rgbaColor=color)

    def _create_inv_index(self, human):
        inv_index = {}
        for j in range(p.getNumJoints(human)):
            info = p.getJointInfo(human, j)
            inv_index[info[1]] = info[0]
        return inv_index

    def addPlaybackTraj(self, traj, human="human1"):
        """ adds a trajectory to the player for playback

        Keyword arguments:
        traj -- playback trajectory
        human -- name of human to add trajectory to (default "human1")
        """
        self._playbackTrajs.append((human, traj))
        if self._start_playback is None or self._start_playback > traj.startframe:
            self._start_playback = traj.startframe
        if self._end_playback is None or self._end_playback < traj.endframe:
            self._end_playback = traj.endframe

    def play(self):
        """ start playback of trajectories """
        start_time = time.time()
        while (p.isConnected()):
            frameReal = self._start_playback + (time.time() - start_time) * self._fps
            frame = int(frameReal)
            if frame >= self._end_playback:
                break
            visibleHumans = []
            for human, traj in self._playbackTrajs:
                if frame >= traj.startframe and frame < traj.endframe:
                    visibleHumans.append(human)
                    trajframe = frame - traj.startframe
                    for key in traj.data_fixed:
                        idx = self._inv_index.get(key, -1)
                        if idx != -1:
                            p.resetJointState(self._humans[human], idx, traj.data_fixed[key])

                    for i in range(len(traj.data[trajframe])):
                        idx = self._inv_index.get(traj.description[i], -1)
                        if idx != -1:
                            p.resetJointState(self._humans[human], idx, traj.data[trajframe][i])

            for human in self._humans:
                if human in visibleHumans:
                    p.resetBasePositionAndOrientation(self._humans[human], [0, 0, 0], [0, 0, 0, 1])
                else:
                    p.resetBasePositionAndOrientation(self._humans[human], [0, 0, -10], [0, 0, 0, 1])  # TODO: is there a cleaner way to hide humans?
