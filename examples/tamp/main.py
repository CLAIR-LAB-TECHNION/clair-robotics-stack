from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import (
    ur5e_1,
    ur5e_2,
)
from clair_robotics_stack.planning.tamp.tamp_runner import (
    TAMPRunner,
    TAMPRunnerCallbacks,
)
from clair_robotics_stack.planning.tamp.up_utils import create_up_problem
from helpers import PickPlaceActionExecutor, PickPlaceStateEstimator, PickPlaceSensors

DOMAIN_FILE = "domain.pddl"
PROBLEM_FILE = "problem.pddl"

# TODO configure camera pose
CAMERA_CONFIG = [0, 0, 0, 0, 0, 0]

# TODO fill out appropriate bounds
LOCATION_BOUNDS = {
    "pos1": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
    "pos2": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
    "pos3": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
    "pos4": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
}

LOCATION_CENTERS = {
    k: [(a + b) / 2 for a, b in zip(v[0], v[1])] for k, v in LOCATION_BOUNDS.items()
}

DEFAULT_BLOCK_CLASSES = ["wooden cube", "wooden block", "wooden box"]

BLOCK_CLASSES = {
    "red": ["red" + s for s in DEFAULT_BLOCK_CLASSES],
    "green": ["green" + s for s in DEFAULT_BLOCK_CLASSES],
    "brown": ["brown" + s for s in DEFAULT_BLOCK_CLASSES],
}

# TODO figure out holding height
HOLDING_HEIGHT = 0.15


class PickPlaceTampCallbacks(TAMPRunnerCallbacks):
    def on_episode_start(self):
        print("Episode started")

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
    state_estimator = PickPlaceStateEstimator(
        ur5e_1["ip"], CAMERA_CONFIG, BLOCK_CLASSES, LOCATION_CENTERS
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
