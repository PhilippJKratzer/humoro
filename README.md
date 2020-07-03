# humoro (Human Motion for Robotics)
This repository contains a human urdf model, a trajectory class to store/load human data and a player to play back the motion data using pybullet. A small example motion file is included. For more motion data in the format check out [our datasets](https://github.com/PhilippJKratzer/mocap-mlr-datasets).


## Install
The repository needs the python packages numpy, pybullet, h5py, PyQt5
```bash
python -m pip install -r requirements.txt
```

## Example
```bash
python examples/play_traj.py datasets/examples/full_body_example.hdf5  # playback example motion data
```
![Pybullet Viewer](doc/screenshots/pybullet1.png?raw=true "Pybullet Viewer")
