import threading
import time
import os
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
from ur_lab.camera.configurations_and_params import depth_fx, depth_fy, depth_ppx, depth_ppy, color_fx, color_fy, color_ppx, \
    color_ppy, depth_to_color_translation
from ur_lab.camera.utils import get_mean_depth


def project_color_pixel_to_depth_pixel(color_image_point, depth_image):
    # the projection depends on depth of the point which we don't know yet.
    # we will take the mean depth of the region around the pixel without projection

    mean_depth = get_mean_depth(depth_image, color_image_point, window_size=20)
    if mean_depth == -1:
        return (-1, -1)  # no depth around that point, cannot project

    x_color = (color_image_point[0] - color_ppx) / color_fx
    y_color = (color_image_point[1] - color_ppy) / color_fy

    color_frame_point = np.array([x_color * mean_depth, y_color * mean_depth, mean_depth])

    depth_frame_point = color_frame_point - depth_to_color_translation

    depth_image_point = [(depth_frame_point[0] * depth_fx / depth_frame_point[2]) + depth_ppx,
                         (depth_frame_point[1] * depth_fy / depth_frame_point[2]) + depth_ppy]

    return depth_image_point


class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_frame_bgr(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image * self.depth_scale
        return color_image, depth_image

    def get_frame_rgb(self):
        bgr, depth = self.get_frame_bgr()
        rgb = None
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, depth

    def plotable_depth(self, depth_image, max_depth=3):
        depth_image = np.clip(depth_image, 0, max_depth)
        depth_image = (depth_image / max_depth * 255).astype(np.uint8)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        return depth_image

    def plot_depth(self, depth_image, max_depth=3):
        plotable_depth = self.plotable_depth(depth_image, max_depth)
        plt.imshow(plotable_depth)
        plt.show()

    def plot_rgb(self, rgb_image):
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.show()


class RealsenseCameraWithRecording(RealsenseCamera):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.record_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def start_recording(self, file_path, max_depth=5, fps=30):
        if self.recording:
            print("Already recording")
            return

        self.recording = True
        self.record_thread = threading.Thread(target=self._record, args=(file_path, max_depth, fps))
        self.record_thread.start()

    def stop_recording(self):
        if not self.recording:
            print("Not recording")
            return

        self.recording = False
        if self.record_thread:
            self.record_thread.join()

    def _record(self, file_path, max_depth, fps):
        directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(directory, exist_ok=True)
        color_path = os.path.join(directory, f"{base_name}_color.mp4")
        depth_path = os.path.join(directory, f"{base_name}_depth.mp4")

        color_video = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
        # fps for depth is low because it's much more jittery and it makes video weight alot
        depth_video = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (1280, 720))

        frame_interval = 1.0 / fps  # Time interval between frames
        last_frame_time = time.time()
        last_depth_frame_time = time.time()

        while self.recording:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                color_frame, depth_frame = super().get_frame_bgr()
                if color_frame is not None and depth_frame is not None:
                    with self.frame_lock:
                        self.latest_frame = (color_frame.copy(), depth_frame.copy())

                    color_video.write(color_frame)
                    if current_time - last_depth_frame_time >= 0.5:  # Only record depth every 0.5 seconds
                        depth_frame_vis = np.clip(depth_frame, 0, max_depth)
                        depth_frame_vis = (depth_frame_vis / max_depth * 255).astype(np.uint8)
                        depth_frame_vis = cv2.cvtColor(depth_frame_vis, cv2.COLOR_GRAY2BGR)
                        depth_video.write(depth_frame_vis)
                        last_depth_frame_time = current_time

                    last_frame_time = current_time
            else:
                time.sleep(0.001)  # Short sleep to prevent busy-waiting

        color_video.release()
        depth_video.release()

    def get_frame_bgr(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame
        return super().get_frame_bgr()


def main():
    camera = RealsenseCameraWithRecording()

    # Start recording
    camera.start_recording("./test_video", max_depth=5, fps=30)

    try:
        for i in range(5):
            # Get a frame once every two seconds
            color_frame, depth_frame = camera.get_frame_rgb()
            if color_frame is not None and depth_frame is not None:
                pass
                plt.imshow(color_frame)
                plt.show()
            else:
                print("Frame not captured")

            time.sleep(2)

    finally:
        camera.stop_recording()

if __name__ == "__main__":
    main()

