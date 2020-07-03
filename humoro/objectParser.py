import h5py
from humoro.trajectory import Trajectory


def parseHDF5(filepath, startframe=0, endframe=-1):
    """ Loads rigid body objects from a hdf5 file containing motion data in our own format

    Keyword arguments:
    filepath -- path to hdf5 file
    startframe -- frame in the file to start reading (default 0)
    endframe -- frame in the file to end reading (default -1)

    Returns list of trajectories
    """
    trajs = []
    f_hdf5 = h5py.File(filepath, 'r')
    keys = f_hdf5['bodies'].keys()
    ids = []
    for bodyname in keys:
        data = f_hdf5['bodies'][bodyname][startframe:endframe]  # open as numpy array
        ids.append(int(bodyname.split()[1]))
        trajs.append(Trajectory(data=data, description=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "rot_w"], startframe=startframe))
    return trajs, ids
