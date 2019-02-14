import os
import sys
import time
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, ".."))
from humoro.trajectory import Trajectory
from humoro.player_pybullet import Player


def main():
    """ Example showing how to display two human models and assigning three short trajectories to them """

    # Open pybullet and spawn two humans
    p = Player()
    p.spawnHuman("Human1")
    p.spawnHuman("Human2", color=[0, .9, .1, 1.])

    # Load human data
    full_data = Trajectory()
    full_data.loadTrajHDF5(_path_file + "/../datasets/examples/full_body_example.hdf5")

    # Create 3 trajectories from data and assign to humans
    traj1 = full_data.subTraj(startframe=0, endframe=1000)
    traj2 = full_data.subTraj(startframe=1600, endframe=2400)
    p.addPlaybackTraj(traj1, "Human1")
    p.addPlaybackTraj(traj2, "Human1")

    traj3 = full_data.subTraj(startframe=0, endframe=2400)
    traj3.startframe = 400  # change start of playback for this trajectory at frame 400
    p.addPlaybackTraj(traj3, "Human2")

    # Start playback
    p.play()
    time.sleep(5)  # don't close immediately after playback finished


if __name__ == '__main__':
    main()
