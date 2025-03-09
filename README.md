# clair-robotics-stack

## Installation:
#### Clone the repository:
```bash
git clone https://github.com/CLAIR-LAB-TECHNION/clair-robotics-stack.git
```

##### Install Poetry
(preferably in a clean virtual environment of python 3.10)
```bash
pip install poetry
```

#### Install dependencies:
```bash
poetry install --no-root
```

## Validating Installations for Lab Urs 
make sure:
* ur5e_2 robot is fully on (the one at the end)
* make sure the robot is on remote control
* make sure you are connected to the robot's network via wifi or ethernet
    the wifi password is on the bottom of the router
* make sure you set up static wifi address for your machine in the robots network.
 It must start with: 192.168.0.xx (xx is a number between 100-255)

run simple example:
```bash
python -m ur_lab.examples.lab_basic_example
```
if you get errors, you can try and ping the robot ip adress. You can see the ip at *ur_lab/robot_inteface/robots_metadata.py*

Some examples are on jupyter notebooks, You should install jupyterlab on your environment to run them.