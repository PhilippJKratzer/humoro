import os
import sys
import time
import h5py
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../.."))
from humoro.trajectory import Trajectory
from humoro.player_pybullet import Player


def loadSegmentFile(path):
    f = h5py.File(path, "r")
    segments = f["segments"][:]
    f.close()
    return segments


def main():
    """ Example displaying reaching trajectories used in paper: Towards Combining Motion Optimization and Data Driven Dynamical Models for Human Motion Prediction; Kratzer, Toussaint, Mainprice; 2018
    Make sure you stored the data file in the right folder."""

    # Load full trajectories from file
    traj = Trajectory()
    traj.loadTrajHDF5(_path_file + "/../../datasets/upper-body/mocap.hdf5")
    segments = loadSegmentFile(_path_file + "/../../datasets/upper-body/segment.hdf5")

    # go through segments and extract reaching trajectories
    trajectories_reaching = []
    playback_idx = 0
    pause = 120  # pause for 1 sec between reaching motions
    for seg in segments:
        print (seg)
        if seg[2].decode() == "reaching" or seg[2].decode() == "reachingL" or seg[2].decode() == "reachingR":
            rtraj = traj.subTraj(int(seg[0]), int(seg[1]))
            rtraj.startframe = playback_idx  # change startframe for playback
            trajectories_reaching.append(rtraj)
            playback_idx += len(rtraj.data) + pause

    print ("Number of reaching trajectories: " + str(len(trajectories_reaching)))

    # Open pybullet and spawn a human
    p = Player(fps=120)
    p.spawnHuman("Human1", upper_body=True, hide=False)

    # add all reaching trajectories to playback list
    for traj in trajectories_reaching:
        p.addPlaybackTraj(traj, "Human1")

    # Start playback
    while True:
        p.play()
        time.sleep(1)


if __name__ == '__main__':
    main()
