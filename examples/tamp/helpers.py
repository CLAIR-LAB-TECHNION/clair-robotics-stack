from collections import defaultdict
from typing import Any

import numpy as np

from clair_robotics_stack.planning.tamp.action_executer import ActionExecuter
from clair_robotics_stack.planning.tamp.state_estimator import ThreeLayerStateEstimator
from clair_robotics_stack.planning.tamp.up_utils import state_dict_to_up_state
from clair_robotics_stack.ur.lab_setup.manipulation.manipulation_controller_2fg import (
    ManipulationController2FG,
)
from clair_robotics_stack.ur.lab_setup.manipulation.robot_with_motion_planning import (
    RobotInterfaceWithMP,
)
from clair_robotics_stack.camera.realsense_camera import RealsenseCameraWithRecording
from clair_robotics_stack.vision.object_detection import ObjectDetection
from clair_robotics_stack.camera.configurations_and_params import (
    color_camera_intrinsic_matrix,
)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class PickPlaceSensors:
    def __init__(self):
        self.camera = RealsenseCameraWithRecording()

    def __call__(self) -> dict:
        out = {}
        out["rgb"], out["depth"] = self.camera.get_frame_rgb()
        return out


class PickPlaceStateEstimator(ThreeLayerStateEstimator):
    def __init__(
        self,
        up_problem,
        camera_robot_ip,
        camera_config,
        block_classes,
        location_bounds,
        holding_height,
        detection_confidence_threshold=0.1,
    ):
        super().__init__()

        self.up_problem = up_problem
        self.cam_bot = RobotInterfaceWithMP.build_from_robot_name_and_ip(camera_robot_ip, 'ur5e_1')

        self.block_classes = block_classes
        all_block_classes = [b for bs in block_classes.values() for b in bs]

        self.location_bounds = location_bounds
        self.holding_height = holding_height

        self.detector = ObjectDetection(
            all_block_classes, min_confidence=detection_confidence_threshold
        )

        # set camera in position
        # self.cam_bot.move_home(speed=1.0)
        self.cam_bot.moveJ(camera_config)

    def estimate_state(self, observations) -> tuple:
        motion_state = self._estimate_motion_state(observations)
        task_state = self._estimate_task_state(motion_state)

        return task_state, motion_state, None

    def _estimate_motion_state(self, observations) -> dict:
        # get observation data
        rgb, depth = observations["rgb"], observations["depth"]

        # detect objects
        bboxes, confidences, results = self.detector.detect_objects([rgb])

        # result is also returned as batch, we have only 1 element in the batch
        bboxes, confidences, results = (
            bboxes[0].cpu().numpy(),
            confidences[0].cpu().numpy(),
            results[0],
        )
        classes = results.boxes.cls.cpu().numpy()
        id_to_class = results.names

        # collect block positions for each block class
        block_pos_dict = defaultdict(list)
        for bbox, cls, conf in zip(bboxes, classes, confidences):
            cls_name = id_to_class[cls]
            block_pos_dict[cls_name].append(self._get_block_pos(bbox, depth))

        motion_state = {}
        for block_name, block_detector_classes in self.block_classes.items():
            potential_block_pos = []
            for cls_name, block_pos in block_pos_dict.items():
                if cls_name in block_detector_classes:
                    potential_block_pos.extend(block_pos_dict[cls_name])
            if potential_block_pos:
                motion_state[block_name] = np.mean(potential_block_pos, axis=0)

        return motion_state

    def _estimate_task_state(self, motion_state):
        """
        Convert motion state to task state using predicates from the PDDL domain.
        Task state is a dictionary of predicates and their values.
        """
        print('motion_state from _estimate_task_state:', motion_state)
        task_state = {}

        # Get all blocks and locations from the motion state and predefined locations
        blocks = list(self.block_classes.keys())
        locations = list(self.location_bounds.keys())

        # initial assumptions before checking predicates
        task_state["handempty()"] = True
        for location in locations:
            task_state[f"occupied({location})"] = False

        # Check at predicate for each block and location
        for block in blocks:
            for location in locations:
                is_at = self._at(block, location, motion_state)
                task_state[f"at({block},{location})"] = is_at

                # If block is at this location, mark it as occupied
                if is_at:
                    task_state[f"occupied({location})"] = True

            # Check if block is being held
            is_holding = self._holding(block, motion_state)
            task_state[f"holding({block})"] = is_holding

            # If any block is being held, hand is not empty
            if is_holding:
                task_state["handempty()"] = False

        # convert to unified-planning state
        return state_dict_to_up_state(self.up_problem, task_state)

    def _get_block_pos(self, bbox, depth):
        block_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        block_center_int = [int(block_center[0]), int(block_center[1])]

        # get the depth at the center of the bounding, average on a small area around the center:
        window_in_depth = depth[
            block_center_int[1] - 2 : block_center_int[1] + 2,
            block_center_int[0] - 2 : block_center_int[0] + 2,
        ]

        # if some depth is zero, this is not a valid pixel:
        window_in_depth = window_in_depth[window_in_depth > 0]
        block_depth = np.mean(window_in_depth)

        # get block pos in camera frame
        fx = color_camera_intrinsic_matrix[0, 0]
        fy = color_camera_intrinsic_matrix[1, 1]
        ppx = color_camera_intrinsic_matrix[0, 2]
        ppy = color_camera_intrinsic_matrix[1, 2]

        x_cam = (block_center[0] - ppx) * block_depth / fx
        y_cam = (block_center[1] - ppy) * block_depth / fy
        z_cam = block_depth
        p_cam = [x_cam, y_cam, z_cam]

        # convert camera frame point to world frame point
        point_world = self.cam_bot.gt.point_camera_to_world(
            p_cam, "ur5e_1", self.cam_bot.getActualQ()
        )

        return point_world

    def _at(self, block: str, location: str, motion_state: dict) -> bool:
        """Predicate: block is at location"""
        if block not in motion_state:
            return False
        block_pos = motion_state[block]
        min_pos, max_pos = self.location_bounds[location]
        return all(min_pos[i] <= block_pos[i] <= max_pos[i] for i in range(3))

    def _holding(self, block: str, motion_state: dict) -> bool:
        """Predicate: robot is holding the block"""
        if block not in motion_state:
            return False
        return bool(motion_state[block][2] > self.holding_height)


class PickPlaceActionExecutor(ActionExecuter):
    def __init__(self, manip_robot_ip, location_centers):
        super().__init__()
        self.manip_robot = ManipulationController2FG.build_from_robot_name_and_ip(
            manip_robot_ip, "ur5e_2"
        )
        self.location_centers = location_centers

    def pick_up(self, b, l):
        block_pos = self._motion_state[b]
        self.manip_robot.pick_up(
            block_pos[0],
            block_pos[1],
            np.pi/2,
            start_height=0.2,
            replan_from_home_if_failed=True,
        )

        return True  #TODO return False if failed

    def put_down(self, b, l):
        location_pos = self.location_centers[l]
        self.manip_robot.put_down(
            location_pos[0],
            location_pos[1],
            np.pi/2,
            start_height=0.15,
            replan_from_home_if_failed=True,
        )

        return True  #TODO return False if failed


class SimulationSensors:
    def __init__(self, env):
        self.env = env

    def __call__(self) -> dict:
        print('enter __call__')
        out = {}
        out["rgb"], out["depth"] = self.env.render()

        # print('out:', out)

        im = Image.fromarray(out["rgb"])
        number = np.random.randint(0, 10000)
        im.save(f'./rgb_frames/frame{number}.png')

        print('saved rgb frame')

        depth_normalized = (out["depth"] - np.min(out["depth"])) / (np.max(out["depth"]) - np.min(out["depth"]))
        plt.imsave(f'./depth_frames/frame{number}.png', np.array(depth_normalized), cmap='gray')

        print('saved dapth frame')

        return out


class SimulationStateEstimator(ThreeLayerStateEstimator):
    def __init__(
        self,
        up_problem,
        # camera_robot_ip,
        # camera_config,
        block_classes,
        location_bounds,
        holding_height,
        detection_confidence_threshold=0.01,
    ):
        super().__init__()

        self.up_problem = up_problem
        # self.cam_bot = RobotInterfaceWithMP.build_from_robot_name_and_ip(camera_robot_ip, 'ur5e_1')

        self.block_classes = block_classes
        all_block_classes = [b for bs in block_classes.values() for b in bs]

        self.location_bounds = location_bounds
        self.holding_height = holding_height

        print('all_block_classes', all_block_classes)

        self.detector = ObjectDetection(
            all_block_classes, min_confidence=detection_confidence_threshold
        )

        # set camera in position
        # self.cam_bot.move_home(speed=1.0)
        # self.cam_bot.moveJ(camera_config)

    def estimate_state(self, observations) -> tuple:
        print('start estimate_state')
        motion_state = self._estimate_motion_state(observations)
        task_state = self._estimate_task_state(motion_state)

        return task_state, motion_state, None

    def _estimate_motion_state(self, observations) -> dict:
        # get observation data
        # print('observations:', observations)
        # rgb, depth = observations["rgb"], observations["depth"]

        # detect objects
        bboxes, confidences, results = self.detector.detect_objects([observations])

        # result is also returned as batch, we have only 1 element in the batch
        bboxes, confidences, results = (
            bboxes[0].cpu().numpy(),
            confidences[0].cpu().numpy(),
            results[0],
        )
        print("bboxes:", bboxes)
        print('confidences:', confidences)
        print('results:', results)
        classes = results.boxes.cls.cpu().numpy()
        id_to_class = results.names

        # collect block positions for each block class
        block_pos_dict = defaultdict(list)
        for bbox, cls, conf in zip(bboxes, classes, confidences):
            cls_name = id_to_class[cls]
            block_pos_dict[cls_name].append(self._get_block_pos(bbox, depth))

        motion_state = {}
        for block_name, block_detector_classes in self.block_classes.items():
            print('block_name:', block_name)
            potential_block_pos = []
            for cls_name, block_pos in block_pos_dict.items():
                if cls_name in block_detector_classes:
                    potential_block_pos.extend(block_pos_dict[cls_name])
            if potential_block_pos:
                motion_state[block_name] = np.mean(potential_block_pos, axis=0)

        #TODO: remove when state estimator will be integrated
        motion_state = {
            'red': [-0.7, -0.6, 0.03],
            'green': [-0.7, -0.7, 0.03],
            'blue': [-0.7, -0.8, 0.03],
        }

        return motion_state

    def _estimate_task_state(self, motion_state):
        """
        Convert motion state to task state using predicates from the PDDL domain.
        Task state is a dictionary of predicates and their values.
        """
        task_state = {}

        # Get all blocks and locations from the motion state and predefined locations
        blocks = list(self.block_classes.keys())
        locations = list(self.location_bounds.keys())

        # initial assumptions before checking predicates
        task_state["handempty()"] = True
        for location in locations:
            task_state[f"occupied({location})"] = False

        # Check at predicate for each block and location
        for block in blocks:
            for location in locations:
                is_at = self._at(block, location, motion_state)
                task_state[f"at({block},{location})"] = is_at

                # If block is at this location, mark it as occupied
                if is_at:
                    task_state[f"occupied({location})"] = True

            # Check if block is being held
            is_holding = self._holding(block, motion_state)
            task_state[f"holding({block})"] = is_holding

            # If any block is being held, hand is not empty
            if is_holding:
                task_state["handempty()"] = False

        # convert to unified-planning state
        return state_dict_to_up_state(self.up_problem, task_state)

    def _get_block_pos(self, bbox, depth):
        block_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        block_center_int = [int(block_center[0]), int(block_center[1])]

        # get the depth at the center of the bounding, average on a small area around the center:
        window_in_depth = depth[
            block_center_int[1] - 2 : block_center_int[1] + 2,
            block_center_int[0] - 2 : block_center_int[0] + 2,
        ]

        # if some depth is zero, this is not a valid pixel:
        window_in_depth = window_in_depth[window_in_depth > 0]
        block_depth = np.mean(window_in_depth)

        # get block pos in camera frame
        fx = color_camera_intrinsic_matrix[0, 0]
        fy = color_camera_intrinsic_matrix[1, 1]
        ppx = color_camera_intrinsic_matrix[0, 2]
        ppy = color_camera_intrinsic_matrix[1, 2]

        x_cam = (block_center[0] - ppx) * block_depth / fx
        y_cam = (block_center[1] - ppy) * block_depth / fy
        z_cam = block_depth
        p_cam = [x_cam, y_cam, z_cam]

        # convert camera frame point to world frame point
        point_world = self.cam_bot.gt.point_camera_to_world(
            p_cam, "ur5e_1", self.cam_bot.getActualQ()
        )

        return point_world

    def _at(self, block: str, location: str, motion_state: dict) -> bool:
        """Predicate: block is at location"""
        if block not in motion_state:
            print(f"Block {block} not in motion state")
            return False
        block_pos = motion_state[block]
        print('block_pos:', block_pos)
        min_pos, max_pos = self.location_bounds[location]
        return all(min_pos[i] <= block_pos[i] <= max_pos[i] for i in range(3))
    #TODO: check taht there is no colflict in the heights, between static ones that i put to expected ones from self
    def _holding(self, block: str, motion_state: dict) -> bool:
        """Predicate: robot is holding the block"""
        if block not in motion_state:
            return False
        return bool(motion_state[block][2] > self.holding_height)


class SimulationMotionExecutor(ActionExecuter):
    def __init__(self, motion_executer, location_centers):
        super().__init__()
        self.motion_executer = motion_executer
        self.location_centers = location_centers

    def pick_up(self, b, l):
        block_pos = self._motion_state[b]
        self.motion_executer.pick_up(
            "ur5e_2",
            block_pos[0],
            block_pos[1],
            start_height=0.2,
        )

        return True  #TODO return False if failed

    def put_down(self, b, l):
        location_pos = self.location_centers[l]
        self.motion_executer.put_down(
            "ur5e_2",
            location_pos[0],
            location_pos[1],
            start_height=0.15,
        )

        return True  #TODO return False if failed