import pybullet_utils.bullet_client as bc
import pybullet
import os
from humoro.pybullet_helpers import create_inv_index
import numpy as np
import time

class HumanKin:
    def __init__(self, urdf=None, upper_body=False):
        self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        if urdf is None:
            if upper_body is True:
                self.human = self.p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human_upper_body.urdf")
            else:
                self.human = self.p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + "/data/human.urdf")
        else:
            self.human = self.p.loadURDF(urdf)
        self.inv_index = create_inv_index(self.human, self.p)
        self.numJoints = self.p.getNumJoints(self.human)

    def set_state_fixed(self, traj):
        idxes = []
        values = []
        for key in traj.data_fixed:
            idx = self.inv_index.get(key, -1)
            if idx != -1:
                idxes.append(idx)
                values.append([traj.data_fixed[key]])
        self.p.resetJointStatesMultiDof(self.human, idxes, values)

    def set_state(self, traj, frameidx):
        self.set_frame(traj.data[frameidx], traj.description)
        self.set_state_fixed(traj)

    def set_frame(self, frame, description):
        idxes = []
        values = []
        for i in range(len(description)):
            idx = self.inv_index.get(description[i], -1)
            if idx != -1:
                idxes.append(idx)
                values.append([frame[i]])
        self.p.resetJointStatesMultiDof(self.human, idxes, values)

    def get_position(self, idx):
        return np.array(self.p.getLinkState(self.human, idx)[0])

    def get_rotation(self, idx):
        link_rot = self.p.getLinkState(self.human, idx)[1]
        return np.reshape(np.array(self.p.getMatrixFromQuaternion(link_rot)), (3,3))

    def get_jacobian(self, idx):
        jointstates = [i[0] for i in self.p.getJointStates(self.human, range(self.p.getNumJoints(self.human)))]
        zero_vec = [0.]*len(jointstates)
        jac = np.array(self.p.calculateJacobian(self.human, idx, [0., 0., 0.], jointstates, zero_vec, zero_vec)[0])
        return jac
