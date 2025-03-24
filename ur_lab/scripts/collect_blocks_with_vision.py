import numpy as np
import typer
from matplotlib import pyplot as plt
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from ur_lab.camera.realsense_camera import RealsenseCamera
from vision.image_block_position_estimator import ImageBlockPositionEstimator
from ur_lab.manipulation.utils import ur5e_2_distribute_blocks_in_workspace_uniform, ur5e_2_collect_blocks_from_positions
from ur_lab.utils.workspace_utils import (workspace_x_lims_default,
                                          workspace_y_lims_default)
from vision.utils import (lookat_verangle_distance_to_robot_config, detections_plots_no_depth_as_image,
                                 detections_plots_with_depth_as_image)


# camera pose:
lookat = [np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default), 0]  # middle of the workspace
lookat[0] += 0.1  # move a bit from the window
lookat[1] += 0.0
verangle = 50
distance = 0.6
y_offset = 0.2

app = typer.Typer()

@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 4,
         use_depth: bool = 1,):

    camera = RealsenseCamera()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController2FG(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 1.75, 1.75
    r2_controller.speed, r2_controller.acceleration = 3.0, 4.0

    r1_sensing_config = lookat_verangle_distance_to_robot_config(lookat, verangle, distance, gt, ur5e_1["name"],
                                                                 y_offset=y_offset)
    if r1_sensing_config is None:
        print("Could not find a valid robot configuration for the camera")
        return
    motion_planner.vis_config(ur5e_1["name"], r1_sensing_config)

    r1_controller.plan_and_move_home(speed=1.5, acceleration=1.5)

    # r2 distribute blocks and clear out
    actual_block_positions = ur5e_2_distribute_blocks_in_workspace_uniform(n_blocks, r2_controller)
    r2_controller.plan_and_move_home(speed=0.5, acceleration=0.5)

    # r1 take image
    r1_controller.plan_and_moveJ(r1_sensing_config)

    im, depth = camera.get_frame_rgb()
    if use_depth:
        positions, annotations = position_estimator.get_block_positions_depth(im, depth, r1_sensing_config)
        plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                       workspace_x_lims_default, workspace_y_lims_default,
                                                       actual_positions=actual_block_positions)
    else:
        positions, annotations = position_estimator.get_block_position_plane_projection(im, r1_sensing_config,
                                                                                        plane_z=-0.02)
        plot_im = detections_plots_no_depth_as_image(annotations[0], annotations[1], positions,
                                                     workspace_x_lims_default, workspace_y_lims_default,
                                                     actual_positions=actual_block_positions)

    # plot in hires:
    plt.figure(figsize=(12, 12), dpi=512)
    plt.imshow(plot_im)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # r1 clear out
    r1_controller.plan_and_move_home(speed=0.5, acceleration=0.5)

    # r2 collect blocks
    ur5e_2_collect_blocks_from_positions(positions, r2_controller)


if __name__ == "__main__":
    app()
