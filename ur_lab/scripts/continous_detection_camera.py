import cv2
import typer
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.robot_inteface.robot_interface import RobotInterface
from ur_lab.robot_inteface.robots_metadata import ur5e_1
from ur_lab.camera.realsense_camera import RealsenseCamera
from vision.image_block_position_estimator import ImageBlockPositionEstimator
from vision.utils import detections_plots_no_depth_as_image, detections_plots_with_depth_as_image



workspace_x_lims = [-0.9, -0.54]
workspace_y_lims = [-1.0, -0.55]

actual_positions = [[-0.6930666821574081, -0.9634592614554912], [-0.5424938100024435, -0.7398593034783345],
                    [-0.722997473205969, -0.8308432570058277], [-0.8432591095761119, -0.8370767468866123],
                    [-0.8783381379793164, -0.5599983546790344]]


app = typer.Typer()
@app.command(
    context_settings={"ignore_unknown_options": True})
def main(use_depth: bool = 1):
    # TODO actual positions as input or from file

    camera = RealsenseCamera()
    gt = GeometryAndTransforms.build()
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims, workspace_y_lims, gt)

    robot = RobotInterface(ur5e_1['ip'])
    robot.freedriveMode(free_axes=[1] * 6)

    while True:
        robot_config = robot.getActualQ()
        bgr, depth = camera.get_frame_bgr()
        if bgr is None or depth is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if use_depth:
            positions, annotations = position_estimator.get_block_positions_depth(rgb, depth, robot_config)
            plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                           workspace_x_lims, workspace_y_lims,
                                                           actual_positions=actual_positions)
            plot_im = cv2.cvtColor(plot_im, cv2.COLOR_RGB2BGR)
            cv2.imshow('detections', plot_im)
            cv2.waitKey(1)

        else:
            positions, annotations = position_estimator.get_block_position_plane_projection(rgb, robot_config)
            plot_im = detections_plots_no_depth_as_image(annotations[0], annotations[1], positions,
                                                         workspace_x_lims, workspace_y_lims,
                                                         actual_positions=actual_positions)
            plot_im = cv2.cvtColor(plot_im, cv2.COLOR_RGB2BGR)
            cv2.imshow('detections', plot_im)
            cv2.waitKey(1)


if __name__ == "__main__":
    app()
