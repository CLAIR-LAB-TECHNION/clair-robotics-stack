import logging
import time
from ur_lab.manipulation.robot_with_motion_planning import RobotInterfaceWithMP
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from motion_planning.motion_planner import MotionPlanner
from ur_lab.robot_inteface.vgc10_gripper import VG10C


class ManipulationControllerVG(RobotInterfaceWithMP):
    def __init__(self, robot_ip, robot_name, motion_palnner: MotionPlanner,
                 geomtry_and_transofms: GeometryAndTransforms, freq=50, gripper_id=0):
        super().__init__(robot_ip, robot_name, motion_palnner, geomtry_and_transofms, freq)

        self.gripper = VG10C(robot_ip, gripper_id)

    @classmethod
    def build_from_robot_name_and_ip(cls, robot_ip, robot_name):
        motion_planner = MotionPlanner()
        geomtry_and_transofms = GeometryAndTransforms(motion_planner)
        return cls(robot_ip, robot_name, motion_planner, geomtry_and_transofms)

    #### Gripper functions ####
    def grasp(self, channel=2, vaccum_power=60, wait_time=0.2):
        logging.debug(f"Grasping on channel {channel} ({self._ip}, vaccum_power: {vaccum_power})")
        res = self.gripper.vg10c_grip(channel, vaccum_power)
        if res != 0:
            logging.warning(f"Failed to grasp ({self._ip})")
        time.sleep(wait_time)

    def release_grasp(self, channel=2, wait_time=0.2):
        logging.debug(f"Releasing grasp on channel {channel} ({self._ip})")
        res = self.gripper.vg10c_release(channel)
        if res != 0:
            logging.warning(f"Failed to release grasp ({self._ip})")
        time.sleep(wait_time)

    def get_vaccum_status(self,):
        """
        return dict with keys: "channelA", "channelB" and vaccume in each. If nothing is grasped it's 0
        """
        return self.gripper.vg10c_get_vacuum()

    #### Pick and place functions ####
    def pick_up(self, x, y, rz, start_height=0.2, channel=2, vaccum_power=60):
        logging.debug(f"Picking up at {x, y, rz} ({self._ip})")

        self.plan_and_move_to_xyzrz(x, y, start_height, rz)
        above_pickup_config = self.getActualQ()
        self.release_grasp(channel)

        logging.debug(f"{self.robot_name} moving down until contact")
        lin_speed = min(self.linear_speed / 2, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])

        # robot retracts after contact, but we need the gripper to touch object firmly
        self.moveL_relative([0, 0, -0.01], speed=0.1)

        self.grasp(channel, vaccum_power)

        height = self.getActualTCPPose()[2]

        # move up
        self.moveL_relative([0, 0, start_height - height], speed=0.1)
        self.update_mp_with_current_config()

    def put_down(self, x, y, rz, start_height=0.2, channel=2):
        logging.debug(f"Putting down at {x, y, rz} ({self._ip})")

        self.plan_and_move_to_xyzrz(x, y, start_height, rz)
        above_drop_config = self.getActualQ()

        logging.debug(f"{self.robot_name} moving down until contact")
        lin_speed = min(self.linear_speed / 2, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])

        self.release_grasp(channel)

        height = self.getActualTCPPose()[2]

        self.moveL_relative([0, 0, start_height- height], speed=0.1)

        self.update_mp_with_current_config()