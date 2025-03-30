from clair_robotics_stack.camera.realsense_camera import RealsenseCamera
from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import ur5e_1
import numpy as np
import cv2
import typer


max_depth = 5
camera_bot_ip = ur5e_1["ip"]


def main(freedrive: bool = False):
    if freedrive:
        from clair_robotics_stack.ur.lab_setup.robot_inteface.robot_interface import RobotInterface
        robot = RobotInterface(camera_bot_ip)
        robot.freedriveMode()
    try:
        camera = RealsenseCamera()
        while True:
            rgb, depth = camera.get_frame_bgr()
            depth = np.clip(depth, 0, max_depth)
            if rgb is not None and depth is not None:
                # scale just for cv2:
                depth = depth / max_depth
                depth = (depth * 255).astype(np.uint8)
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

                cv2.imshow('image', rgb)
                cv2.imshow('depth', depth)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

    if freedrive:
        robot.endFreedriveMode()


if __name__ == "__main__":
    typer.run(main)

