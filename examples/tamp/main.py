from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import (
    ur5e_1,
    ur5e_2,
)
from clair_robotics_stack.planning.tamp.tamp_runner import (
    TAMPRunner,
    TAMPRunnerCallbacks,
)
from clair_robotics_stack.planning.tamp.up_utils import create_up_problem, get_object_names_dict
from helpers import PickPlaceActionExecutor, PickPlaceStateEstimator, PickPlaceSensors
import matplotlib.pyplot as plt

DOMAIN_FILE = "domain.pddl"
PROBLEM_FILE = "problem.pddl"

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
    "pos1": ([-0.63, -0.55, -0.1], [-0.58, -0.45, 0.1]),
    "pos2": ([-0.81, -0.55, -0.1], [-0.71, -0.45, 0.1]),
    "pos3": ([-0.63, -0.75, -0.1], [-0.58, -0.65, 0.1]),
    "pos4": ([-0.81, -0.75, -0.1], [-0.71, -0.65, 0.1]),
}

LOCATION_CENTERS = {
    k: [(a + b) / 2 for a, b in zip(v[0], v[1])] for k, v in LOCATION_BOUNDS.items()
}

DEFAULT_BLOCK_CLASSES = ["cube", "block", "box", "wooden cube", "wooden block", "wooden box"]

# TODO figure out holding height
HOLDING_HEIGHT = 0.15


class PickPlaceTampCallbacks(TAMPRunnerCallbacks):
    def on_episode_start(self):
        print("Episode started")

    def on_state_update(self, observations):
        print("State updated")
        det_res = self.runner.state_estimator.detector.detect_objects([observations["rgb"]])[-1][0]
        plt.figure()
        out = self.runner.state_estimator.detector.get_annotated_images(det_res)
        plt.imshow(out)
        plt.show()
        print("task_state:\n", self.runner.cur_task_state)
        print("motion_state:\n", self.runner.cur_motion_state)

    def on_replan(self):
        print("Replanning")

    def on_action_start(self, action):
        print(f"Starting action: {action}")

    def on_action_end(self, action, success):
        print(f"Action {action} ended with success: {success}")

    def on_episode_end(self):
        print("Episode ended")


def main():
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


if __name__ == "__main__":
    main()
