import h5py
from humoro.trajectory import Trajectory

def load_gaze(file):
    with h5py.File(file, "r") as f:
        data = f['gaze'][:]
        calib = f['gaze'].attrs['calibration']
    gaze_traj = Trajectory(data=data[:, 2:5], data_fixed={"calibration": calib})
    print(gaze_traj.data.shape)
    return gaze_traj
        
