import io
from PIL import Image
from klampt.math import se3
from matplotlib import pyplot as plt
from ur_lab.motion_planning.geometry_and_transforms import GeometryAndTransforms
import numpy as np
from ur_lab.camera.configurations_and_params import color_camera_intrinsic_matrix
import logging

from ur_lab.motion_planning.motion_planner import MotionPlanner


def project_points_to_image(points, gt: GeometryAndTransforms, robot_name, robot_config,
                            intrinsic_matrix=color_camera_intrinsic_matrix):
    """
    Project points from the world coordinates to the image coordinates
    :param points: points to project nx3 array
    :param gt: geometric transforms object
    :param robot_name: robot with the camera
    :param robot_config: configuration of the robot with the camera
    :return: projected points in the image coordinates nx2 array
    """
    points = np.array(points)
    if points.shape == (3,):  # only one point
        points = points.reshape(1, 3)

    assert points.shape[1] == 3, "points should be nx3 array"

    world_2_camera = gt.world_to_camera_transform(robot_name, robot_config)
    world_2_camera = gt.se3_to_4x4(world_2_camera)

    points_homogenous = np.ones((4, points.shape[0]))
    points_homogenous[:3] = points.T

    # points are column vectors as required
    points_camera_frame_homogenous = world_2_camera @ points_homogenous  # 4x4 @ 4xn = 4xn
    points_camera_frame = points_camera_frame_homogenous[:3]
    points_image_homogenous = intrinsic_matrix @ points_camera_frame  # 3x3 @ 3xn = 3xn
    points_image = points_image_homogenous / points_image_homogenous[2]  # normalize

    return points_image[:2].T


def crop_workspace(image,
                   robot_config,
                   gt: GeometryAndTransforms,
                   workspace_limits_x,
                   workspace_limits_y,
                   z=-0.0,
                   robot_name="ur5e_1",
                   extension_radius=0.04,
                   intrinsic_matrix=color_camera_intrinsic_matrix):
    """
    crop the workspace that is within given workspace limits and return the cropped image and coordinates of
     the cropped image in the original image
    :param image: the image to crop
    :param robot_config: configuration of the robot with the camera, ur5e_1 if not specified
    :param gt: geometric transforms object
    :param workspace_limits_x: workspace max and min limits in x direction in world coordinates
    :param workspace_limits_y: workspace max and min limits in y direction in world coordinates
    :param robot_name: robot with the camera
    :param extension_radius: how much to extend the workspace. half of the box can be outside the workspace limits, thus
     we need to extend at least by half box which is 2cm, but due to noice and inaccuracies, we extend more as default
    :return: cropped image, xyxy within the original image
    """
    extended_x_lim = [workspace_limits_x[0] - extension_radius, workspace_limits_x[1] + extension_radius]
    extended_y_lim = [workspace_limits_y[0] - extension_radius, workspace_limits_y[1] + extension_radius]
    z_lim = [z - extension_radius, z + extension_radius]

    corners = np.array([[extended_x_lim[0], extended_y_lim[0], z_lim[0]],
                        [extended_x_lim[0], extended_y_lim[0], z_lim[1]],
                        [extended_x_lim[0], extended_y_lim[1], z_lim[0]],
                        [extended_x_lim[0], extended_y_lim[1], z_lim[1]],
                        [extended_x_lim[1], extended_y_lim[0], z_lim[0]],
                        [extended_x_lim[1], extended_y_lim[0], z_lim[1]],
                        [extended_x_lim[1], extended_y_lim[1], z_lim[0]],
                        [extended_x_lim[1], extended_y_lim[1], z_lim[1]]])

    corners_image = project_points_to_image(corners, gt, robot_name, robot_config, intrinsic_matrix)
    corners_image = corners_image.astype(int)
    x_min, y_min = np.min(corners_image, axis=0)
    x_max, y_max = np.max(corners_image, axis=0)

    # in case the workspace is out of the image:
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, [x_min, y_min, x_max, y_max]


def detections_plots_no_depth_as_image(cropped_image, orig_image, pred_positions,
                                       ws_lim_x, ws_lim_y, actual_positions=None):
    cropped_image = np.asarray(cropped_image).astype(np.uint8)
    orig_image = np.asarray(orig_image).astype(np.uint8)

    fig, axs = plt.subplots(3, 1, figsize=(5, 10))
    axs[0].imshow(cropped_image)
    axs[1].imshow(orig_image)

    if len(pred_positions) == 0:
        # dummy plot:
        pred_positions = [[0, 0]]
    pred_positions = np.array(pred_positions)
    axs[2].scatter(pred_positions[:, 0], pred_positions[:, 1], c="red", label="Predicted")
    if actual_positions is not None:
        actual_positions = np.array(actual_positions)
        axs[2].scatter(actual_positions[:, 0], actual_positions[:, 1], c="b", label="Actual")
    axs[2].legend()
    extended_x_lim_for_plot = [ws_lim_x[0] - 0.1, ws_lim_x[1] + 0.1]
    extended_y_lim_for_plot = [ws_lim_y[0] - 0.1, ws_lim_y[1] + 0.1]
    axs[2].set_xlim(extended_x_lim_for_plot)
    axs[2].set_ylim(extended_y_lim_for_plot)

    fig.tight_layout()

    # instead of showing, return as image:
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)
    plt.close(fig)

    return img_np


def detections_plots_with_depth_as_image(cropped_image, orig_image, depth_image, pred_positions,
                                         ws_lim_x, ws_lim_y, actual_positions=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(cropped_image)
    axs[0, 1].imshow(orig_image)
    axs[1, 0].imshow(depth_image)

    if len(pred_positions) == 0:
        # dummy plot:
        pred_positions = [[0, 0]]
    pred_positions = np.array(pred_positions)
    if actual_positions is not None:
        actual_positions = np.array(actual_positions)
        axs[1, 1].scatter(actual_positions[:, 0], actual_positions[:, 1], c="b", label="Actual")
    axs[1, 1].scatter(pred_positions[:, 0], pred_positions[:, 1], c="r", label="Predicted")
    extended_x_lim_for_plot = [ws_lim_x[0] - 0.1, ws_lim_x[1] + 0.1]
    extended_y_lim_for_plot = [ws_lim_y[0] - 0.1, ws_lim_y[1] + 0.1]
    axs[1, 1].set_xlim(extended_x_lim_for_plot)
    axs[1, 1].set_ylim(extended_y_lim_for_plot)

    fig.tight_layout()

    # instead of showing, return as image:
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)
    plt.close(fig)

    return img_np


def lookat_verangle_distance_to_camera_transform(lookat, vertical_angle, distance, y_offset=0.3, up_vector=(1, 0, 0)):
    """
    returns the camera se3 transform given the lookat point, vertical angle and distance.
    the camera will be in the same x as the lookat point, and y will be lookat[1] + y_offset
    up_vector is the up vector of the camera, it fixes the free dof that is left given the other params.
    default is [1, 0, 0] which is good because it's toward the workspace, but it can be changed if no solution found
    :return:
    """
    vertical_angle = np.deg2rad(vertical_angle)

    # Calculate the camera position in the world frame
    delta_x = distance * np.cos(vertical_angle)
    delta_z = distance * np.sin(vertical_angle)
    camera_position = np.array([lookat[0] + delta_x, lookat[1] + y_offset, lookat[2] + delta_z])

    # Calculate the direction vector from the camera to the look-at point
    direction = lookat - camera_position
    direction /= np.linalg.norm(direction)  # Normalize the direction vector

    # build rotation matrix from up, right, forward vectors
    # Assume the up vector is [1, 0, 0] for simplicity this is good because it's toward the workspace,
    # the camera will be aligned
    up = up_vector

    # Calculate the right vector
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)  # Normalize the right vector

    # Recalculate the up vector to ensure orthogonality
    up = np.cross(direction, right)

    # Create the rotation matrix, this is the world to camera rotation matrix
    rotation_matrix = np.eye(3)
    rotation_matrix[:, 0] = right
    rotation_matrix[:, 1] = up
    rotation_matrix[:, 2] = direction
    # Invert the rotation matrix to get the camera to world rotation matrix
    rotation_matrix = np.linalg.inv(rotation_matrix)

    return rotation_matrix.flatten(), camera_position


def lookat_verangle_distance_to_robot_config(lookat, vertical_angle, distance, gt, robot_name, y_offset=0.3,
                                             max_trials=50):
    """
    returns the robot configuration given the lookat point, vertical angle and distance.
    the camera will be in the same x as the lookat point, and y will be lookat[1] + y_offset
    or none if no solution is found. This doesn't consider collisions! # TODO find collision free
    """
    trails_left = max_trials
    up_vector = (1, 0, 0)
    config = None

    logging.debug(f"Finding robot config for camera at lookat {lookat}"
                  f" with vertical angle {vertical_angle} and distance {distance}")

    while config is None and trails_left > 0:
        trails_left -= 1

        camera_transform = lookat_verangle_distance_to_camera_transform(lookat, vertical_angle, distance, y_offset,
                                                                        up_vector)
        ee_transform = se3.mul(gt.camera_to_ee_transform(), camera_transform)
        config = gt.motion_planner.ik_solve(robot_name, ee_transform)
        if config is not None and gt.motion_planner.is_config_feasible(robot_name, config) is False:
            config = None

        # sample new up vector, should be normalized
        up_vector = np.random.rand(3)
        up_vector /= np.linalg.norm(up_vector)

    if config is None:
        logging.error(f"Could not find a valid robot configuration for the camera at lookat {lookat}"
                      f" with vertical angle {vertical_angle} and distance {distance} after {max_trials} trials")
    else:
        logging.debug(f"Found robot config for camera at lookat {lookat}"
                      f" with vertical angle {vertical_angle} and distance {distance} after"
                      f" {max_trials - trails_left} trials")

    return config


def lookat_verangle_horangle_distance_to_camera_transform(lookat, vertical_angle, horizontal_angle, distance,
                                                          rotation_angle=0):
    """
    returns the camera se3 transform given the lookat point, vertical angle, horizontal angle and distance.
    the camera will look at the lookat point with the given angles and distance.
    vertical angle is the angle from the xy plane, horizontal angle is the angle from xz plane.
    There's one degree of freedom which is the rotation of the camera around lookat direction
    """
    vertical_angle = np.deg2rad(vertical_angle)
    horizontal_angle = np.deg2rad(horizontal_angle)
    rotation_angle = np.deg2rad(rotation_angle)

    # Calculate the camera position in the world frame
    delta_x = distance * np.cos(vertical_angle) * np.cos(horizontal_angle)
    delta_y = distance * np.cos(vertical_angle) * np.sin(horizontal_angle)
    delta_z = distance * np.sin(vertical_angle)
    assert np.abs(delta_x ** 2 + delta_y ** 2 + delta_z ** 2 - distance ** 2) < 1e-5

    camera_position = np.array([lookat[0] + delta_x, lookat[1] + delta_y, lookat[2] + delta_z])

    # Calculate the direction vector from the camera to the look-at point
    direction = lookat - camera_position
    direction /= np.linalg.norm(direction)  # Normalize the direction vector

    # build rotation matrix from up, right, forward vectors
    forward = direction
    up = np.array([np.cos(rotation_angle), np.sin(rotation_angle), 0])
    up /= np.linalg.norm(up)  # Normalize the up vector
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)  # Normalize the right vector
    up = np.cross(forward, right)

    # Create the rotation matrix, this is the world to camera rotation matrix
    rotation_matrix = np.eye(3)
    rotation_matrix[:, 0] = right
    rotation_matrix[:, 1] = up
    rotation_matrix[:, 2] = forward
    # Invert the rotation matrix to get the camera to world rotation matrix
    rotation_matrix = np.linalg.inv(rotation_matrix)

    return rotation_matrix.flatten(), camera_position


def lookat_verangle_horangle_distance_to_robot_config(lookat, vertical_angle, horizontal_angle,
                                                      distance, gt, robot_name, camera_rotations_to_try=27):
    camera_rotations = np.linspace(0, 270, camera_rotations_to_try)
    for rot in camera_rotations:
        camera_transform = lookat_verangle_horangle_distance_to_camera_transform(lookat, vertical_angle,
                                                                                 horizontal_angle, distance,
                                                                                 rotation_angle=rot)
        ee_transform = se3.mul(gt.camera_to_ee_transform(), camera_transform)
        config = gt.motion_planner.ik_solve(robot_name, ee_transform)
        if config is not None and gt.motion_planner.is_config_feasible(robot_name, config):
            return config

    return None
