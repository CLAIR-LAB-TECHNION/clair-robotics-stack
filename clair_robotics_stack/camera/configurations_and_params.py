import numpy as np
from ..camera.utils import load_camera_params
import os

# extrinsic camera parameters, relative to ee, no rotation just translation
camera_in_ee_orig = np.array([0, -0.105, 0.0395 - 0.15])
camera_in_ee_experimental_correction = np.array([-0.04, -0.01, -0.005])
camera_in_ee = camera_in_ee_orig + camera_in_ee_experimental_correction


# intrinsic camera parameters, and extrinsic just between depth and color camera
# params file path should be relative to the file that is calling this
path_to_params = os.path.join(os.path.dirname(__file__), 'camera_params.json')
camera_params = load_camera_params(path_to_params)
depth_intr = camera_params['depth_intrinsics']
color_intr = camera_params['color_intrinsics']
depth_to_color_extr = camera_params['depth_to_color_extrinsics']

depth_fx = depth_intr['fx']
depth_fy = depth_intr['fy']
depth_ppx = depth_intr['ppx']
depth_ppy = depth_intr['ppy']

color_fx = color_intr['fx']
color_fy = color_intr['fy']
color_ppx = color_intr['ppx']
color_ppy = color_intr['ppy']

color_camera_intrinsic_matrix = np.array([[color_fx, 0, color_ppx],
                                          [0, color_fy, color_ppy],
                                          [0, 0, 1]])

depth_camera_intrinsic_matrix = np.array([[depth_fx, 0, depth_ppx],
                                          [0, depth_fy, depth_ppy],
                                          [0, 0, 1]])

depth_to_color_translation = np.array(depth_to_color_extr['translation'])


if __name__ == '__main__':
    print("fx, fy, ppx, ppy")
    print(depth_fx, depth_fy, depth_ppx, depth_ppy)

