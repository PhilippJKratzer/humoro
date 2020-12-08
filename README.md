# humoro (Human Motion for Robotics)
This repository contains a human urdf model, a trajectory class to store/load human data and a player to play back the motion data using pybullet. A small example motion file is included. For more motion data in the format check out [our datasets](https://github.com/PhilippJKratzer/mocap-mlr-datasets).

## Install
The python requirements can be installed using pip:
```bash
python3 -m pip install -r requirements.txt
```

Some parts of the code depend on pyQt5 and thus Qt5 needs to be installed to run them. For Ubuntu it can be installed with:
```bash
sudo apt install qt5-default
```

## Example
Check out our [getting started notebook](https://github.com/PhilippJKratzer/humoro/blob/master/examples/getting_started.ipynb), which contains detailed installation instructions and explains how to use this repository with our [mogaze dataset](https://humans-to-robots-motion.github.io/mogaze/).

![Mogaze Dataset](https://raw.githubusercontent.com/humans-to-robots-motion/mogaze/master/images/im2.png "Mogaze Dataset")
