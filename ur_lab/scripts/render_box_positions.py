import json

import numpy as np
from matplotlib import pyplot as plt

from ur_lab.vision.state_renderer import PyrenderRenderer


def load_data():
    metadata = json.load(open("../vision/images_data_merged_hires/merged_metadata.json", "r"))
    robot_configs = []
    images = []
    depth_images = []
    actual_block_positions = []

    for im_metadata in metadata:
        robot_configs.append(im_metadata["ur5e_1_config"])

        image_path = f"../vision/images_data_merged_hires/images/{im_metadata['image_rgb']}"
        image = np.load(image_path)
        images.append(image)

        depth_images.append(np.load(f"../vision/images_data_merged_hires/depth/{im_metadata['image_depth']}"))

    # actual block positions are the same for all images, so we can just take the first one
    actual_block_positions = metadata[0]["block_positions"]

    return robot_configs, images, depth_images, actual_block_positions


if __name__ == "__main__":
    robot_configs, images, depth_images, actual_block_positions = load_data()
    box_3d_positons = [(p[0], p[1], 0.01) for p in actual_block_positions]
    r = PyrenderRenderer()

    r.set_boxes_positions(box_3d_positons)

    for robot_config, image in zip(robot_configs, images):
        rendered_color, depth = r.render_from_robot_config("ur5e_1", robot_config)

        mask = rendered_color[:, :, 3] > 0  # Mask where alpha > 0
        overlay = np.zeros_like(rendered_color)
        # Extract the green boxes and black edges
        overlay[mask] = rendered_color[mask]

        # plot rendered_color on top of image, treat transparent background of rendered as transparent
        plt.imshow(image)
        plt.imshow(overlay, alpha=0.5)
        plt.axis('off')
        plt.show()
