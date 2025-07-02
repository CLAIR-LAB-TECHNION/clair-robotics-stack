from mujoco_simulation.mujoco_env.sim_env import SimEnv
from mujoco_simulation.motion_planning.motion_executor import MotionExecutor

import mediapy as media
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# TODO: check the limits  
"""
workspace_x_lims = [-1.0, -0.45]
workspace_y_lims = [-1.0, -0.45]
"""

# Locate an area where both robots can reach
# You can choose any location in the area to place the blocks

block_position = [
    [-0.7, -0.6, 0.03],
    [-0.7, -0.7, 0.03],
    [-0.7, -0.8, 0.03],
    [-0.7, -0.9, 0.03]]

# Create the simulation environment and the executor
print("declaretion of env")
render_mode = 'depth_array'
env = SimEnv(render_mode=render_mode)
print("declaretion of executor")
executor = MotionExecutor(env)

# Add blocks to the world by enabling the randomize argument and setting the block position in the reset function of the SimEnv class 
env.reset(randomize=False, block_positions=block_position)
frames = []
# moveJ is utilized when the robot's joints are clear to you but use carefully because there is no planning here
move_to = [1.305356658502026, -0.7908733209856437, 1.4010098471710881, 4.102251451313659, -1.5707962412281837, -0.26543967541515895]
executor.moveJ("ur5e_2", move_to)
frames.append(env.render())

executor.pick_up("ur5e_2", -0.7, -0.6, 0.15)
frames.append(env.render())

executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.7, 0.15])
frames.append(env.render())

executor.put_down("ur5e_2", -0.7, -0.7, 0.20)
frames.append(env.render())

print('here')
print('frames shape:', np.array(frames).shape)
print('here2')
framerate = 60
print('here3')
if render_mode == 'rgb_array':
  for i, image in enumerate(frames):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    im = Image.fromarray(image)
    im.save(f'./depth_frames/frame{i}.png')
else:
  for i, image in enumerate(frames):
    depth_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imsave(f'./depth_frames/frame{i}.png', np.array(depth_normalized), cmap='gray')
# media.show_video(frames, fps=framerate)
print('here4')

executor.wait(4)

# framerate = 60
# media.show_video(frames, fps=framerate)
# for i, image in enumerate(frames):
#   im = Image.fromarray(image)
#   im.save(f'./video_frames/frame{i}.png')












