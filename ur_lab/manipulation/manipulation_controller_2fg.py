from typing import Sequence
from scipy.spatial.transform import Rotation as R
import numpy as np
from ur_lab.manipulation.robot_with_motion_planning import RobotInterfaceWithMP, to_canonical_config
from ur_lab.robot_inteface.robot_interface import RobotInterfaceWithGripper, home_config
from ur_lab.motion_planning.motion_planner import MotionPlanner
from ur_lab.motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.robot_inteface.twofg7_gripper import TwoFG7
from ur_lab.robot_inteface.vgc10_gripper import VG10C
from ur_lab.utils import logging_util
import time
import logging
import chime


class ManipulationController2FG(RobotInterfaceWithMP):
    def __init__(self, robot_ip, robot_name, motion_palnner: MotionPlanner,
                 geomtry_and_transofms: GeometryAndTransforms, freq=50, gripper_id=0):
        super().__init__(robot_ip, robot_name, motion_palnner, geomtry_and_transofms, freq)

        self.gripper = TwoFG7(robot_ip, gripper_id)

        self.min_width = self.gripper.twofg_get_min_external_width()
        self.max_width = self.gripper.twofg_get_max_external_width()

    @classmethod
    def build_from_robot_name_and_ip(cls, robot_ip, robot_name):
        motion_planner = MotionPlanner()
        geomtry_and_transofms = GeometryAndTransforms(motion_planner)
        return cls(robot_ip, robot_name, motion_planner, geomtry_and_transofms)

    #### Gripper functions ####
    def set_gripper(self, width, force, speed, wait_time=0.5):
        logging.debug(f"Setting gripper ({self._ip}), width: {width}, force: {force}, speed: {speed}")
        res = self.gripper.twofg_grip_external(width, force, speed)
        if res != 0:
            logging.warning(f"Failed to set gripper ({self._ip}), width: {width}, force: {force}, speed: {speed}")
        time.sleep(wait_time)

    def grasp(self, wait_time=0.5, force=20, speed=100, width=None):
        if width is None:
            width = self.min_width
        logging.debug(f"Grasping ({self._ip}), min_width: {self.min_width}")
        res = self.gripper.twofg_grip_external(width, force, speed)
        if res != 0:
            logging.warning(f"Failed to grasp ({self._ip})")
        time.sleep(wait_time)

    def release_grasp(self, wait_time=0.5, force=20, speed=100):
        logging.debug(f"Releasing grasp ({self._ip}), max_width: {self.max_width}")
        res = self.gripper.twofg_int_release(self.gripper.max_int_width)
        if res != 0:
            logging.warning(f"Failed to release grasp ({self._ip})")
            print("FAIL")
        time.sleep(wait_time)

    def is_object_gripped(self):
        return self.gripper.twofg_get_grip_detected()

    #### Utils ####

    def approach_vector_to_axis_angle(self, approach_vector, ee_rz):
        mat_z_col = approach_vector
        world_z = np.array([0, 0, 1])

        # Project world Z onto plane perpendicular to approach vector
        proj = world_z - (np.dot(world_z, mat_z_col) * mat_z_col)
        if np.linalg.norm(proj) < 1e-5:
            mat_y_col = np.array([0, -1, 0])
        else:
            mat_y_col = -proj / np.linalg.norm(proj)  # flip Y axis

        mat_x_col = np.cross(mat_y_col, mat_z_col)

        rotation_matrix = np.array([mat_x_col, mat_y_col, mat_z_col]).T
        rotation_matrix = rotation_matrix @ R.from_euler("z", ee_rz).as_matrix()
        axis_angle = R.from_matrix(rotation_matrix).as_rotvec()
        return axis_angle

    #### Pick and place functions ####

    def pick_up_at_angle(self, point: Sequence[float], start_offset: Sequence[float], ee_rz=0,
                         grasp_force=20, grasp_speed=100, grasp_width=None):
        """
        pick up by grasping object at point, by first arriving to the point + offset,
        ee facing the point, rotated around z axis by ee_rz (z axis is ee forward).
        The robot will move from the offset to the point using linear motion, then retract back to the offset.
        If there's no z offset, there will be 1 cm z offset to avoid scratching the surface.
        """
        point = np.asarray(point)
        start_offset = np.asarray(start_offset)

        offset_position = point + start_offset

        approach_vector = -start_offset
        approach_vector =  approach_vector /np.linalg.norm(approach_vector)

        axis_angle = self.approach_vector_to_axis_angle(approach_vector, ee_rz)

        pick_up_start_pose = offset_position.tolist() + axis_angle.tolist()

        pick_up_start_config = self.find_ik_solution(pick_up_start_pose, max_tries=50, for_down_movement=False)
        if pick_up_start_config is None:
            logging.error(f"{self.robot_name} Could not find IK solution for pick up start pose")
            print("\033[93m Could not find IK solution, not moving. \033[0m")
            chime.error()
            return

        pick_up_start_config = to_canonical_config(pick_up_start_config)

        self.release_grasp()

        res = self.plan_and_moveJ(pick_up_start_config)
        if not res:
            print("\033[93m Could not find path, not moving. \033[0m")
            chime.error()
            return
        self.moveL(point.tolist() + axis_angle.tolist(), speed=self.linear_speed,
                   acceleration=self.linear_acceleration)
        self.grasp(force=grasp_force, speed=grasp_speed, width=grasp_width)

        if np.abs(start_offset[2]) <= 0.005:
            # add 0.5 cm z offset to avoid scratching the surface
            offset_position[2] += 0.01
        self.moveL(offset_position.tolist() + axis_angle.tolist(), speed=0.05)

    def put_down_at_angle(self, point: Sequence[float], start_offset: Sequence[float], ee_rz=0,):
        """
        put down by first arriving to the point + offset, ee facing the point, rotated around z axis by ee_rz.
        The robot will move from the offset to the point using linear motion, then retract back to the offset.
        if there's no z offset, there will be 1 cm z offset to avoid scratching the surface.
        """
        point = np.asarray(point)
        start_offset = np.asarray(start_offset)
        add_z_offset = np.abs(start_offset[2]) < 0.005

        offset_position = point + start_offset

        approach_vector = -start_offset
        approach_vector = approach_vector / np.linalg.norm(approach_vector)

        axis_angle = self.approach_vector_to_axis_angle(approach_vector, ee_rz)

        offset_position[2] += 0.01 * int(add_z_offset)
        put_down_start_pose = offset_position.tolist() + axis_angle.tolist()

        put_down_start_config = self.find_ik_solution(put_down_start_pose, max_tries=50, for_down_movement=False)
        if put_down_start_config is None:
            logging.error(f"{self.robot_name} Could not find IK solution for put down start pose")
            print("\033[93m Could not find IK solution, not moving. \033[0m")
            chime.error()
            return
        put_down_start_config = to_canonical_config(put_down_start_config)

        res = self.plan_and_moveJ(put_down_start_config)
        if not res:
            print("\033[93m Could not find path, not moving. \033[0m")
            chime.error()
            return
        self.moveL(point.tolist() + axis_angle.tolist(), speed=self.linear_speed,
                   acceleration=self.linear_acceleration)
        self.release_grasp()
        self.moveL(offset_position.tolist() + axis_angle.tolist(), speed=0.05)


    def pick_up(self, x, y, rz, start_height=0.2, replan_from_home_if_failed=True):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} picking up at {x}{y}{rz} with start height {start_height}")

        # move above pickup location:
        res = self.plan_and_move_to_xyzrz(x, y, start_height, rz)

        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_move_to_xyzrz(x, y, start_height, rz)
            if not res:
                chime.error()
                return

        above_pickup_config = self.getActualQ()
        self.release_grasp()

        # move down until contact, here we move a little bit slower than drop and sense
        # because the gripper rubber may damage from the object at contact:
        logging.debug(f"{self.robot_name} moving down until contact")
        lin_speed = min(self.linear_speed / 2, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])

        # retract one more centimeter to avoid gripper scratching the surface:
        self.moveL_relative([0, 0, 0.01],
                            speed=0.1,
                            acceleration=0.1)
        logging.debug(f"{self.robot_name} grasping and picking up")
        # close gripper:
        self.grasp()
        # move up:
        self.moveJ(above_pickup_config, speed=self.speed, acceleration=self.acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()


    def put_down(self, x, y, rz, start_height=0.2, replan_from_home_if_failed=True):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} putting down at {x}{y}{rz} with start height {start_height}")
        # move above dropping location:
        res = self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
            if not res:
                chime.error()
                return

        above_drop_config = self.getActualQ()

        logging.debug(f"{self.robot_name} moving down until contact to put down")
        # move down until contact:
        lin_speed = min(self.linear_speed, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # release grasp:
        self.release_grasp()
        # back up 10 cm in a straight line :
        self.moveL_relative([0, 0, 0.1], speed=self.linear_speed, acceleration=self.linear_acceleration)
        # move to above dropping location:
        self.moveJ(above_drop_config, speed=self.speed, acceleration=self.acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

    # def sense_height(self, x, y, start_height=0.2):
    #     """
    #     TODO
    #     :param x:
    #     :param y:
    #     :param start_height:
    #     :return:
    #     """
    #     logging.info(f"{self.robot_name} sensing height not tilted! at {x}{y} with start height {start_height}")
    #     self.grasp()
    #
    #     # move above sensing location:
    #     self.plan_and_move_to_xyzrz(x, y, start_height, 0, speed=self.speed, acceleration=self.acceleration)
    #     above_sensing_config = self.getActualQ()
    #
    #     lin_speed = min(self.linear_speed, 0.1)
    #     # move down until contact:
    #     self.moveUntilContact(xd=[0, 0, lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    #     # measure height:
    #     height = self.getActualTCPPose()[2]
    #     # move up:
    #     self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)
    #
    #     # update the motion planner with the new configuration:
    #     self.update_mp_with_current_config()
    #
    #     return height

    def sense_height_tilted(self, x, y, start_height=0.15, replan_from_home_if_failed=True):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} sensing height tilted at {x}{y} with start height {start_height}")
        self.grasp()

        # set end effector to be the tip of the finger
        self.setTcp([0.02, 0.012, 0.15, 0, 0, 0])

        logging.debug(f"moving above sensing point with TCP set to tip of the finger")

        # move above point with the tip tilted:
        pose = self.gt.get_tilted_pose_6d_for_sensing(self.robot_name, [x, y, start_height])
        goal_config = self.find_ik_solution(pose, max_tries=50)
        res = self.plan_and_moveJ(goal_config)

        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return -1

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_moveJ(goal_config)
            if not res:
                chime.error()
                return -1

        above_sensing_config = self.getActualQ()

        logging.debug(f"moving down until contact with TCP set to tip of the finger")

        # move down until contact:
        lin_speed = min(self.linear_speed, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # measure height:
        pose = self.getActualTCPPose()
        height = pose[2]
        # move up:
        self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)

        # set back tcp:
        self.setTcp([0, 0, 0.150, 0, 0, 0])
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

        logging.debug(f"height measured: {height}, TCP pose at contact: {pose}")

        return height
