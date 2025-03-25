import json

import numpy as np


def load_camera_params(filename):
    with open(filename, "r") as f:
        params = json.load(f)
    return params


def get_mean_depth(depth_image, color_pixel, window_size=10):
    x, y = color_pixel
    x, y = int(x), int(y)
    half_window = window_size // 2

    # Define the region of interest (ROI)
    roi = depth_image[max(0, y-half_window):min(depth_image.shape[0], y+half_window),
                      max(0, x-half_window):min(depth_image.shape[1], x+half_window)]

    # Get valid depth values (non-zero)
    valid_depths = roi[roi > 0]

    if len(valid_depths) == 0:
        return -1

    mean_depth = np.mean(valid_depths)
    return mean_depth