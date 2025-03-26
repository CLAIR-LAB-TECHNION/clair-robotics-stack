from clair_robotics_stack.planning.tamp.action_executer import ActionExecuter
from clair_robotics_stack.planning.tamp.state_estimator import ThreeLayerStateEstimator
from clair_robotics_stack.planning.tamp.tamp_runner import TAMPRunner, TAMPRunnerCallbacks
from clair_robotics_stack.ur.lab_setup.manipulation.manipulation_controller_2fg import ManipulationController2FG
from clair_robotics_stack.ur.lab_setup.robot_inteface.robot_interface import RobotInterface
from clair_robotics_stack.camera.realsense_camera import RealsenseCamera
from clair_robotics_stack.vision.object_detection import ObjectDetection

location_bounds = {
    "pos1": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
    "pos2": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
    "pos3": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
    "pos4": ([0.5, 0.5, 0.5], [0.6, 0.6, 0.6]),
}

location_centers = {
    k: [(a + b) / 2 for a, b in zip(v[0], v[1])]
    for k, v in location_bounds.items()
}

DEFAULT_BLOCK_CLASSES = ['wooden cube', 'wooden block', 'wooden box']

block_classes = {
    "red": ["red" + s for s in DEFAULT_BLOCK_CLASSES],
    "green": ["green" + s for s in DEFAULT_BLOCK_CLASSES],
    "brown": ["brown" + s for s in DEFAULT_BLOCK_CLASSES],
}
all_block_classes = [b for bs in block_classes.values() for b in bs]



class PickPlaceActionExecutor(ActionExecuter):
    def __init__(self, manip_robot_ip, camera_robot_ip):
        super().__init__()
        self.manip_robot = ManipulationController2FG.build_from_robot_name_and_ip(manip_robot_ip, "manip-bot")
        self.detector = ObjectDetection(all_block_classes)

    def pick_up(self, b, l):
        block_pose = self._motion_state[b]

    def put_down(self, b, l):
        pass
