import os
import sys
import time
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, ".."))
from humoro.trajectory import Trajectory
from humoro.player_pybullet import Player


def main():
    """ Example for playing a full trajectory file passed by sys.argv """

    # Load the passed trajectory from file
    traj1 = Trajectory()
    traj1.loadTrajHDF5(sys.argv[1])

    # Open pybullet and spawn a human
    p = Player(fps=120)
    p.spawnHuman("Human1")

    p.addPlaybackTraj(traj1, "Human1")

    # Start playback
    while True:
        p.play()
        time.sleep(1)


if __name__ == '__main__':
    main()
