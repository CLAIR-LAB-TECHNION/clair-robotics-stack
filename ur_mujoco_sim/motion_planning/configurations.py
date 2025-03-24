import numpy as np

# constraint joint limits for faster planning. joint 2 is base rotation, don't need entire 360,
# joint 3 is shoulder lift, outside the limits will probably result table collision anyway.
# joint 4 has 2 pi range anyway. all the others can do fine with 3pi range
limits_l = [0, -1.5 * np.pi, -4.5, -np.pi, -3 * np.pi / 2, -3 * np.pi / 2, -3 * np.pi / 2, 0]
limits_h = [0, 1.5 * np.pi, 1., np.pi, 3 * np.pi / 2, 3 * np.pi / 2, 3 * np.pi / 2, 0]

default_config = {
    # "type": "lazyrrg*",
    "type": "rrt*",
    "bidirectional": True,
    "connectionThreshold": 30.0,
    # "shortcut": True, # only for rrt
}

block_size = [0.04, 0.04, 0.04]
# blocks are configured at spear_env/assets/scenes/3tableblocksworld/scene.xml
# Right now this is a constant and can't be changed from here
# Note that sizes in the xml file are half, because boxes are defined by their center and half size

robot_height = 0.903 + 0.163 - 0.089159
# 0.903 is the height of the robot mount, 0.163 is the height of the shift of shoulder link in mujoco,
# 0.089159 is the height of shoulder link in urdf for klampt
mount_top_base = robot_height - 0.01  # avoid collision between robot base and mount

table_size = [0.6, 0.6, 0.01]
table_left_pos = [0.0, -0.6, 0.7]
table_right_pos = [0.0, 0.6, 0.7]
table_front_pos = [0.6, 0.0, 0.7]

# relative position of grasped object from end effector
grasp_offset = 0.02
