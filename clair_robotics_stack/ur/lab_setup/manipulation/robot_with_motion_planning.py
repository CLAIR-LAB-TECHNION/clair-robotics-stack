import logging
import time
import numpy as np
from ....planning.motion.geometry_and_transforms import GeometryAndTransforms
from ....planning.motion.motion_planner import MotionPlanner
from ..robot_inteface.robot_interface import RobotInterface, home_config
from ..utils import logging_util


def canninical_last_joint_config(config):
    while config[5] > np.pi:
        config[5] -= 2 * np.pi

    while config[5] < -np.pi:
        config[5] += 2 * np.pi

    return config


def to_valid_limits_config(config):
    for i in range(6):
        while config[i] >= 2 * np.pi:
            config[i] -= 2 * np.pi

        while config[i] <= - 2 * np.pi:
            config[i] += 2 * np.pi

    return config


class RobotInterfaceWithMP(RobotInterface):
    """
    Extension for the RobotInterfaceWithGripper with motion planning and geometry.
    """
    # those are angular in radians:
    speed = 1.0
    acceleration = 1.0

    # and this is linear, ratio that makes sense:
    @property
    def linear_speed(self):
        return self.speed * 0.1

    @property
    def linear_acceleration(self):
        return self.acceleration * 0.1

    def __init__(self, robot_ip, robot_name, motion_palnner: MotionPlanner,
                 geomtry_and_transofms: GeometryAndTransforms, freq=50, visualize=False):
        super().__init__(robot_ip, freq)

        logging_util.setup_logging()

        self.robot_name = robot_name
        self.motion_planner = motion_palnner
        self.gt = geomtry_and_transofms

        # Add window name to distinguish between different visualizations
        if visualize and not MotionPlanner.vis_initialized:
            motion_palnner.visualize(window_name="robots_visualization")

        self.setTcp([0, 0, 0.150, 0, 0, 0])

        if visualize:
            motion_palnner.visualize()
            time.sleep(0.2)

    @classmethod
    def build_from_robot_name_and_ip(cls, robot_ip, robot_name):
        motion_planner = MotionPlanner()
        geomtry_and_transofms = GeometryAndTransforms(motion_planner)
        return cls(robot_ip, robot_name, motion_planner, geomtry_and_transofms)

    def update_mp_with_current_config(self):
        self.motion_planner.update_robot_config(self.robot_name, self.getActualQ())
        logging.info(f"{self.robot_name} Updated motion planner with current configuration {self.getActualQ()}")

    def move_path(self, path, speed=None, acceleration=None, blend_radius=0.02, asynchronous=False):
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        super().move_path(path, speed=speed, acceleration=acceleration,
                          blend_radius=blend_radius, asynchronous=asynchronous)


    def find_ik_solution(self, pose, max_tries=10, for_down_movement=True, shoulder_constraint_for_down_movement=0.3):
        """
        if for_down_movement is True, there will be a heuristic check that tha shoulder is not facing down, so when
        movel will be called it won't collide with the table when movingL down.
        """
        # try to find the one that is closest to the current configuration:
        solution = self.getInverseKinematics(pose)
        if solution == []:
            logging.error(f"{self.robot_name} no inverse kinematic solution found at all "
                          f"for pose {pose}")

        def is_safe_config(q):
            if for_down_movement:
                safe_shoulder = -shoulder_constraint_for_down_movement > q[1] > -np.pi + shoulder_constraint_for_down_movement
                safe_for_sensing_close = True
                # if 0 > pose[1] > -0.4 and -0.1 < pose[0] < 0.1:  # too close to robot base
                #     print(pose)
                #     safe_for_sensing_close = -3*np.pi/4 < q[0] < -np.pi/2 or np.pi/2 < q[0] < 3*np.pi/4
                return safe_shoulder and safe_for_sensing_close
            else:
                return True

        trial = 1
        while ((self.motion_planner.is_config_feasible(self.robot_name, solution) is False or
               is_safe_config(solution) is False)
               and trial < max_tries):
            trial += 1
            # try to find another solution, starting from other random configurations:
            qnear = np.random.uniform(-np.pi / 2, np.pi / 2, 6)
            solution = self.getInverseKinematics(pose, qnear=qnear)

        solution = canninical_last_joint_config(solution)
        solution = to_valid_limits_config(solution)

        if trial == max_tries:
            logging.error(f"{self.robot_name} Could not find a feasible IK solution after {max_tries} tries")
            return None
        elif trial > 1:
            logging.info(f"{self.robot_name} Found IK solution after {trial} tries")
        else:
            logging.info(f"{self.robot_name} Found IK solution in first try")

        return solution

    def plan_and_moveJ(self, q, speed=None, acceleration=None, visualise=True):
        """
        Plan and move to a joint configuration.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        start_config = self.getActualQ()

        logging.info(f"{self.robot_name} planning and movingJ to {q} from {start_config}")

        if visualise:
            self.motion_planner.vis_config(self.robot_name, q, vis_name="goal_config",
                                           rgba=(0, 1, 0, 0.5))
            self.motion_planner.vis_config(self.robot_name, start_config,
                                           vis_name="start_config", rgba=(1, 0, 0, 0.5))

        # plan until the ratio between length and distance is lower than 2, but stop if 8 seconds have passed
        path = self.motion_planner.plan_from_start_to_goal_config(self.robot_name,
                                                                  start_config,
                                                                  q,
                                                                  max_time=30,
                                                                  max_length_to_distance_ratio=2)

        if path is None:
            logging.error(f"{self.robot_name} Could not find a path")
            print("Could not find a path, not moving.")
            return False
        else:
            logging.info(f"{self.robot_name} Found path with {len(path)} waypoints, moving...")

        if visualise:
            self.motion_planner.vis_path(self.robot_name, path)

        self.move_path(path, speed, acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()
        return True

    def plan_and_move_home(self, speed=None, acceleration=None):
        """
        Plan and move to the home configuration.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        self.plan_and_moveJ(home_config, speed, acceleration)

    def plan_and_move_to_xyzrz(self, x, y, z, rz, speed=None, acceleration=None, visualise=True,
                               for_down_movement=True):
        """
        if for_down_movement is True, there will be a heuristic check that tha shoulder is not facing down, so when
        movel will be called it won't collide with the table when movingL down.
        Plan and move to a position in the world coordinate system, with gripper
        facing downwards rotated by rz.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        target_pose_robot = self.gt.get_gripper_facing_downwards_6d_pose_robot_frame(self.robot_name,
                                                                                     [x, y, z],
                                                                                     rz)
        logging.info(f"{self.robot_name} planning and moving to xyzrz={x}{y}{z}{rz}. "
                     f"pose in robot frame:{target_pose_robot}")

        shoulder_constraint = 0.15 if z < 0.2 else 0.35
        goal_config = self.find_ik_solution(target_pose_robot, max_tries=50, for_down_movement=for_down_movement,)
        return self.plan_and_moveJ(goal_config, speed, acceleration, visualise)
        # motion planner is automatically updated after movement


def to_canonical_config(config, tol=0, ignore_joints=(False,) *6):
    """
    change config to be between -pi and pi for all joints with tolerance of tol
    """
    for i in range(6):
        if ignore_joints[i]:
            continue
        while config[i] > np.pi + tol:
            config[i] -= 2 * np.pi
        while config[i] < -np.pi - tol:
            config[i] += 2 * np.pi

    return config
