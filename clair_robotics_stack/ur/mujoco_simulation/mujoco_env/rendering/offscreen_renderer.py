import mujoco
import numpy as np


class OffscreenRenderer:
    def __init__(self, model, data, camera_id, width=1280, height=720, depth=False, segmentation=False):
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.depth = depth
        self.segmentation = segmentation

        # Initialize MuJoCo renderer
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)

        if segmentation:
            self.renderer.enable_segmentation_rendering()

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render()
        if self.depth:
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            depth = np.expand_dims(depth, axis=-1)
            self.renderer.disable_depth_rendering()
            return np.concatenate((rgb, depth), axis=-1)
        elif self.segmentation:
            segmentation = self.renderer.render()
            return segmentation
        else:
            return rgb

    def close(self):
        # No need to explicitly close the renderer in newer MuJoCo versions
        pass
