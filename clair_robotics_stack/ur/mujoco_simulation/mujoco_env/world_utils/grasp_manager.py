import numpy as np
from .object_manager import ObjectManager
from .configurations_and_constants import *


class GraspManager:
    def __init__(self, mj_model, mj_data, object_manager: ObjectManager, min_grasp_distance=0.15):
        self._mj_model = mj_model
        self._mj_data = mj_data
        self.object_manager = object_manager
        self.min_grasp_distance = min_grasp_distance

        self.graspable_objects_names = object_manager.object_names

        # Only considers one robot (we can expand it to a dictionary)
        self._ee_mj_data = self._mj_data.body('robot_1_ur5e/robot_1_adhesive gripper/')

        self.attached_object_name = None

    def grasp_block_if_close_enough(self) -> bool:
        """
        find the nearest object and grasp it if it is close enough
        """
        object_positions = [self.object_manager.get_object_pos(name) for name in self.graspable_objects_names]
        gripper_position = self._ee_mj_data.xpos

        # a block is grasped if distance in x and distance in y are less than 0.015 and in z less than 0.03
        for i, object_position in enumerate(object_positions):
            if np.abs(object_position[0] - gripper_position[0]) < 0.015 and \
                    np.abs(object_position[1] - gripper_position[1]) < 0.015 and\
                    np.abs(object_position[2] - gripper_position[2]) < 0.05:
                self.grasp_object(self.graspable_objects_names[i])
                return True

        return False

    def grasp_object(self, object_name):
        """
        attatch this object to the gripper position
        """
        self.attached_object_name = object_name
        self.update_grasped_object_pose()

    def release_object(self):
        """
        release the object from the gripper
        """
        self.attached_object_name = None

    def update_grasped_object_pose(self):
        """
        update the pose of the object that is currently grasped to be on the gripper
        """
        if self.attached_object_name is None:
            return

        target_position = self._ee_mj_data.xpos
        target_orientation = self._ee_mj_data.xquat
        target_velocities = self._ee_mj_data.cvel
        target_velocities = np.zeros(6)

        # add shift to target position to make sure object is a bit below end effector, but in ee frame
        target_position_in_ee = np.array([0, 0, grasp_offset])
        target_position = target_position + self._ee_mj_data.xmat.reshape(3, 3) @ target_position_in_ee

        self.object_manager.set_object_pose(self.attached_object_name, target_position, target_orientation)
        self.object_manager.set_object_vel(self.attached_object_name, target_velocities)
