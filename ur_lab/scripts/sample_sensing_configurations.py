import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from klampt.math import se3
from klampt import vis
import time

from vision.utils import lookat_verangle_distance_to_camera_transform

app = typer.Typer()


def visualize_robot(motion_planner, gt, lookat, vertical_angle, distance, y_offset=0.3,):
    cam_transform = lookat_verangle_distance_to_camera_transform(lookat, vertical_angle, distance, y_offset)
    vis.add("cam", cam_transform)

    ee_transform = se3.mul(gt.camera_to_ee_transform(), cam_transform)
    config = motion_planner.ik_solve("ur5e_1", ee_transform)

    if config is not None:
        motion_planner.vis_config("ur5e_1", config)
    else:
        print("No solution found")


@app.command(
    context_settings={"ignore_unknown_options": True})
def main():
    workspace_limits_x = [-0.9, -0.54]
    workspace_limits_y = [-1.0, -0.55]
    workspace_center = [(workspace_limits_x[0] + workspace_limits_x[1]) / 2,
                        (workspace_limits_y[0] + workspace_limits_y[1]) / 2,
                        0]
    workspace_corners = [[workspace_limits_x[0], workspace_limits_y[0], 0.02],
                         [workspace_limits_x[1], workspace_limits_y[0], 0.02],
                         [workspace_limits_x[1], workspace_limits_y[1], 0.02],
                         [workspace_limits_x[0], workspace_limits_y[1], 0.02]]

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    motion_planner.visualize(beckend="PyQt5")
    for i, corner in enumerate(workspace_corners):
        motion_planner.show_point_vis(corner, f"c{i}")

    cam_transform = lookat_verangle_distance_to_camera_transform(workspace_center, 45, 0.5)
    vis.add("cam", cam_transform)
    ee_transform = se3.mul(gt.camera_to_ee_transform(), cam_transform)

    config = motion_planner.ik_solve("ur5e_1", ee_transform)

    motion_planner.vis_config("ur5e_1", config)
    # TODO what about collisions!
    # TODO assert that
    time.sleep(60)


if __name__ == "__main__":
    app()

