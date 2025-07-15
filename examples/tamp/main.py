from clair_robotics_stack.ur.mujoco_simulation.motion_planning.motion_executor import MotionExecutor
from clair_robotics_stack.ur.mujoco_simulation.mujoco_env.sim_env import SimEnv
import argparse

from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import (
    ur5e_1,
    ur5e_2,
)
from clair_robotics_stack.planning.tamp.tamp_runner import (
    TAMPRunner,
    TAMPRunnerCallbacks,
)
from clair_robotics_stack.planning.tamp.up_utils import create_up_problem, get_object_names_dict
from helpers import PickPlaceActionExecutor, PickPlaceStateEstimator, PickPlaceSensors, SimulationMotionExecutor, SimulationSensors, SimulationStateEstimator
import matplotlib.pyplot as plt
from clair_robotics_stack.planning.tamp.state_estimator import ThreeLayerStateEstimator

import os
import cv2

DOMAIN_FILE = "./examples/tamp/domain.pddl"
PROBLEM_FILE = "./examples/tamp/problem.pddl"

# TODO configure camera pose
CAMERA_ROBOT_CONFIG = [
    -1.1932991186725062,
    -2.102870603600973,
    -1.6199605464935303,
    -1.772470613519186,
    0.5902572274208069,
    2.3439011573791504
]

# TODO fill out appropriate bounds
LOCATION_BOUNDS = {
    "pos1": ([-0.75, -0.65, -0.1], [-0.65, -0.55, 0.1]),
    "pos2": ([-0.75, -0.75, -0.1], [-0.65, -0.65, 0.1]),
    "pos3": ([-0.75, -0.85, -0.1], [-0.65, -0.75, 0.1]),
    "pos4": ([-0.75, -0.35, -0.1], [-0.71, -0.25, 0.1]),
}

LOCATION_CENTERS = {
    k: [(a + b) / 2 for a, b in zip(v[0], v[1])] for k, v in LOCATION_BOUNDS.items()
}

DEFAULT_BLOCK_CLASSES = ["cube", "block", "box","rectangle", "wooden cube", "wooden block", "wooden box", "wooden rectangle"]

# TODO figure out holding height
HOLDING_HEIGHT = 0.15


class PickPlaceTampCallbacks(TAMPRunnerCallbacks):
    def on_episode_start(self):
        print("Episode started")

    def on_state_update(self, observations):
        print("State updated")
        print("observations['rgb'].shape:", observations['rgb'].shape)
        det_res = self.runner.state_estimator.detector.detect_objects([observations["rgb"]])[-1][0]
        # plt.figure()
        out = self.runner.state_estimator.detector.get_annotated_images(det_res)
        # plt.imshow(out)
        # plt.show()
        print("task_state from on_state_update:\n", self.runner.cur_task_state)
        print("motion_state from on_state_update:\n", self.runner.cur_motion_state)

    def on_replan(self):
        print("Replanning")

    def on_action_start(self, action):
        print(f"Starting action: {action}")

    def on_action_end(self, action, success):
        print(f"Action {action} ended with success: {success}")

    def on_episode_end(self):
        print("Episode ended")


def main(use_simulation):
    if not use_simulation:
        print("here")
        problem = create_up_problem(DOMAIN_FILE, PROBLEM_FILE)
        executer = PickPlaceActionExecutor(ur5e_2["ip"], LOCATION_CENTERS)

        block_names = get_object_names_dict(problem)['block']
        block_classes = {
            block_name: [block_name + " " + s for s in DEFAULT_BLOCK_CLASSES]
            for block_name in block_names
        }

        state_estimator = PickPlaceStateEstimator(
            problem, ur5e_1["ip"], CAMERA_ROBOT_CONFIG, block_classes, LOCATION_BOUNDS, HOLDING_HEIGHT,
            detection_confidence_threshold=0.05
        )
        sensor_fn = PickPlaceSensors()

        callbacks = PickPlaceTampCallbacks()
        runner = TAMPRunner(
            problem=problem,
            executer=executer,
            state_estimator=state_estimator,
            sensor_fn=sensor_fn,
            callbacks=callbacks,
        )
        runner.run_episode()
    else:
        #TODO: add usage in executer and planner for the mujoco simulation
        print("Simulation mode is not implemented yet.")
        problem = create_up_problem(DOMAIN_FILE, PROBLEM_FILE)
        print('declaration of env')
        env = SimEnv()
        print('declaration of executor')
        motion_executer = MotionExecutor(env)
        executer = SimulationMotionExecutor(motion_executer, LOCATION_CENTERS)
        print('declaration of block_names')
        block_names = get_object_names_dict(problem)['block']
        block_classes = {
            block_name: [block_name + " " + s for s in DEFAULT_BLOCK_CLASSES]
            for block_name in block_names
        }
        print('declaration of state_estimator')
        state_estimator = SimulationStateEstimator(problem, block_classes, LOCATION_BOUNDS, HOLDING_HEIGHT,
            detection_confidence_threshold=0.05)


        # rgb_path = "./clair_robotics_stack/ur/rgb_frames"
        # depth_path = "./clair_robotics_stack/ur/depth_frames" 
        # rgb_images = []
        # depth_images = []

        # for rgb_frame in os.listdir(rgb_path):
        #     # Load and upscale the image
        #     image_path = os.path.join(rgb_path, rgb_frame)
        #     rgb_image = cv2.imread(image_path)
        #     rgb_images.append(rgb_image)

        # for depth_frame in os.listdir(depth_path):
        #     # Load and upscale the image
        #     image_path = os.path.join(rgb_path, depth_frame)
        #     depth_image = cv2.imread(image_path)
        #     depth_images.append(depth_image)

        # # sensor_fn = #TODO
        # observations = {
        #     "rgb": rgb_images,
        #     "depth": depth_images
        # }

        sensor_fn = SimulationSensors(env)

        callbacks = PickPlaceTampCallbacks()
        runner = TAMPRunner(
            problem=problem,
            executer=executer,
            state_estimator=state_estimator,
            sensor_fn=sensor_fn,
            callbacks=callbacks,
        )
        runner.run_episode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolean argument example")
    parser.add_argument('--use_simulation', action='store_true', help="Enable debug mode")
    args = parser.parse_args()
    main(args.use_simulation)
