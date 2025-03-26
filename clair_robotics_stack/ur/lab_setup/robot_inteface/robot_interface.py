from rtde_control import RTDEControlInterface as rtdectrl
from rtde_receive import RTDEReceiveInterface as rtdercv
from .twofg7_gripper import TwoFG7
from numpy import pi
import time
import logging

from .vgc10_gripper import VG10C

home_config = [0, -pi/2, 0, -pi/2, 0, 0]


# class RobotInterface(rtdectrl, rtdeio, rtdercv):
class RobotInterface(rtdectrl, rtdercv):
    def __init__(self, robot_ip, freq=50):
        rtdectrl.__init__(self, robot_ip, freq)
        rtdercv.__init__(self, robot_ip, freq)
        # rtdeio.__init__(self, robot_ip, freq)
        self._ip = robot_ip

    def move_home(self, speed=0.2, acceleration=0.2, asynchronous=False):
        logging.debug(f"Moving to home position ({self._ip}), speed: {speed}, acceleration: {acceleration}")
        self.moveJ(q=home_config, speed=speed, acceleration=acceleration, asynchronous=asynchronous)

    def move_path(self, path, speed=0.5, acceleration=0.5, blend_radius=0.05, asynchronous=False):
        logging.debug(f"Moving along path ({self._ip}),"
                      f" speed: {speed}, acceleration: {acceleration}, blend_radius: {blend_radius}")
        path_with_params = [[*target_config, speed, acceleration, blend_radius] for target_config in path]
        # last section should have blend radius 0 otherwise the robot will not reach the last target
        path_with_params[-1][-1] = 0
        self.moveJ(path_with_params, asynchronous=asynchronous)

    def moveL_relative(self, relative_position, speed=0.5, acceleration=0.5, asynchronous=False):
        target_pose = self.getActualTCPPose()
        for i in range(3):
            target_pose[i] += relative_position[i]
        logging.debug(f"Moving to relative position ({self._ip}),"
                      f"from {self.getActualTCPPose()} to {target_pose}, speed: {speed}, acceleration: {acceleration}")

        self.moveL(target_pose, speed, acceleration, asynchronous)

class RobotInterfaceWithGripper(RobotInterface):
    def __init__(self, robot_ip, freq=50, gripper_id=0):
        raise DeprecationWarning("This class is deprecated, use RobotInterfaceWith2FG7 instead")
        super().__init__(robot_ip, freq)
        self.gripper = TwoFG7(robot_ip, gripper_id)

        self.min_width = self.gripper.twofg_get_min_external_width()
        self.max_width = self.gripper.twofg_get_max_external_width()

    def set_gripper(self, width, force, speed, wait_time=0.5):
        logging.debug(f"Setting gripper ({self._ip}), width: {width}, force: {force}, speed: {speed}")
        res = self.gripper.twofg_grip_external(width, force, speed)
        if res != 0:
            logging.warning(f"Failed to set gripper ({self._ip}), width: {width}, force: {force}, speed: {speed}")
        time.sleep(wait_time)

    def grasp(self, wait_time=0.5):
        logging.debug(f"Grasping ({self._ip}), min_width: {self.min_width}")
        res = self.gripper.twofg_grip_external(self.min_width, 20, 100)
        if res != 0:
            logging.warning(f"Failed to grasp ({self._ip})")
        time.sleep(wait_time)

    def release_grasp(self, wait_time=0.5):
        logging.debug(f"Releasing grasp ({self._ip}), max_width: {self.max_width}")
        res = self.gripper.twofg_grip_external(self.max_width, 20, 100)
        if res != 0:
            logging.warning(f"Failed to release grasp ({self._ip})")
        time.sleep(wait_time)

    def is_object_gripped(self):
        return self.gripper.twofg_get_grip_detected()


class RobotInterfaceWithSuctionGripper(RobotInterface):
    def __init__(self, robot_ip, freq=50, suction_id=0):
        super().__init__(robot_ip, freq)
        self.suction = VG10C(robot_ip, suction_id)

