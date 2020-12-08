# humoro (Human Motion for Robotics)
This repository contains a human urdf model, a trajectory class to store/load human data and a player to play back the motion data using pybullet. A small example motion file is included. For more motion data in the format check out [our datasets](https://github.com/PhilippJKratzer/mocap-mlr-datasets).

## Install
The repository needs the python packages numpy, pybullet, h5py, PyQt5
```bash
python3 -m pip install -r requirements.txt
```

## Example
Check out our [getting started notebook](https://github.com/PhilippJKratzer/humoro/blob/master/examples/getting_started.ipynb) that explains how to use this repository with our [mogaze dataset](https://humans-to-robots-motion.github.io/mogaze/).
