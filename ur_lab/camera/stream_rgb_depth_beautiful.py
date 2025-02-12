from ur_lab.camera.realsense_camera import RealsenseCamera
import numpy as np
import cv2


max_depth = 5


if __name__ == "__main__":
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

