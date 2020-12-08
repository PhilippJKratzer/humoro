import os
import sys
import time
_path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_path_file, "../.."))
from humoro.trajectory import Trajectory
from humoro.player_pybullet import Player
import argparse
import humoro.load_scenes as load_scenes
from humoro.gaze import load_gaze

def main():
    """ Example for playing a full trajectory file passed by sys.argv """
    parser = argparse.ArgumentParser()
    parser.add_argument('traj', type=str, help='trajectory file to playback')
    parser.add_argument('--obj', type=str, help='object file to playback', default=None)
    parser.add_argument('--gaze', type=str, help='gaze file to playback', default=None)
    parser.add_argument('--segfile', type=str, help='segmentation file to display', default=None)
    parser.add_argument('--scene', type=str, help='file specification for the scene', default=None)
    parser.add_argument('--nocontrols', type=bool, help='disable the small timeline window to avoid using pyQt5', default=False)
    parser.add_argument('--fps', type=int, default=120,
                        help='framerate for playback')
    args = parser.parse_args()

    p = Player(fps=args.fps)

    # Load the passed trajectory from file
    traj1 = Trajectory()
    traj1.loadTrajHDF5(args.traj)

    # Load objects
    if args.obj != None:
        load_scenes.autoload_objects(p, args.obj, args.scene)

    if args.gaze != None:
        p.addPlaybackTrajGaze(load_gaze(args.gaze))


    p.spawnHuman("Human1")
    p.addPlaybackTraj(traj1, "Human1")

    # Start playback
    if args.nocontrols:
        p.play()
    else:
        p.play_controls(args.segfile)



if __name__ == '__main__':
    main()
