import random
import mujoco
import numpy as np
from typing import Dict, List
import re

class ObjectManager:
    """convenience class to manage graspable objects in the mujoco simulation"""

    def __init__(self, mj_model, mj_data):
        self._mj_model = mj_model
        self._mj_data = mj_data

        # manipulated objects have 6dof free joint that must be named in the mcjf.
        all_joint_names = [self._mj_model.joint(i).name for i in range(self._mj_model.njnt)]
        # print('all_joint_names:', all_joint_names)

        # all bodies that ends with "box"
        prefixs = ("can", "block", "bread", "lemon", "bottle", "milk", "cereal")
        self.object_names = [name for name in all_joint_names if name.startswith(prefixs)]
        self.objects_mjdata_dict = {name: self._mj_model.joint(name) for name in self.object_names}
        self.initial_positions_dict = self.get_all_block_positions()
        self.workspace_x_lims = [-0.9, -0.54]
        self.workspace_y_lims = [-1.0, -0.55]
        self.block_size = .04

    def reset(self, randomize=True, block_positions=None):
        """
        Reset the object positions in the simulation.
        Args:
            randomize: if True, randomize the positions of the blocks, otherwise set them to initial positions.
        """

        def check_block_collision(new_pos):
            """Tests if new position for block collides with any other block"""
            for pos in block_positions:
                pos_np = np.array(pos)
                if np.linalg.norm(new_pos - pos_np) < 2 * self.block_size:
                    return True
            block_positions.append(list(new_pos))
            return False

        if randomize:
            # randomize block positions
            block_positions = []
            for _ in range(len(self.object_names)):
                # generate random position for block
                block_location = [random.uniform(*self.workspace_x_lims), random.uniform(*self.workspace_y_lims), 0.05]
                # check if block collides with any other previous new block position
                while check_block_collision(np.array(block_location)):
                    # generate new random position for block
                    block_location = [random.uniform(*self.workspace_x_lims), random.uniform(*self.workspace_y_lims),
                                      0.05]
            # set blocks to new positions
            print('block_positions in reset:', block_positions)
            self.set_all_block_positions(block_positions)
        else:
            if block_positions:
                self.set_all_block_positions(block_positions)
            else:
                # print('self.initial_positions_dict:', self.initial_positions_dict)
                # self.set_all_block_positions(list(self.initial_positions_dict.values()))
                if all(isinstance(item, np.ndarray) for item in self.initial_positions_dict):
                    positions = [list(pos) for pos in self.initial_positions_dict]
                    # print('positions:', positions)
                    self.set_all_block_positions(positions)
                else:
                    # print('not only blocks scinenario, setting all items positions')
                    self.set_all_items_positions(self.initial_positions_dict)


    def get_object_pos(self, name: str):
        return self._mj_data.joint(name).qpos[:3]

    def set_object_pose(self, name: str, pos, quat):
        joint_id = self.objects_mjdata_dict[name].id
        pos_adr = self._mj_model.jnt_qposadr[joint_id]
        self._mj_data.qpos[pos_adr:pos_adr + 7] = np.concatenate([pos, quat])

    def set_object_vel(self, name: str, cvel):
        joint_id = self.objects_mjdata_dict[name].id
        vel_adr = self._mj_model.jnt_dofadr[joint_id]
        self._mj_data.qvel[vel_adr:vel_adr + 6] = cvel

    def get_block_position_from_mj_id(self, block_id: int) -> np.ndarray:
        """
        Get the position of a block in the simulation.
        Args:
            block_id: the id of the block to get the position of.
        Returns:
            the position of the block in format [x, y ,z].
        """
        return self._mj_data.joint(block_id).qpos[:3]

    def get_all_block_positions_dict(self) -> Dict[str, np.ndarray]:
        """
        Get the positions of all blocks in the simulation.
        Returns:
            a dictionary of block names to their positions, positions will be in format {name: [x, y ,z], ...}.
        """
        return {name: self.get_block_position_from_mj_id(self.objects_mjdata_dict[name].id) for name in self.object_names}

    def get_all_block_positions(self) -> List[np.ndarray]:
        """
        Get the positions of all blocks in the simulation.
        Returns:
            a dictionary of block names to their positions, positions will be in format {name: [x, y ,z], ...}.
        """
        all_blocks = all(s.startswith("block") for s in self.object_names)
        if all_blocks:
            return [self.get_block_position_from_mj_id(self.objects_mjdata_dict[name].id) for name in self.object_names]
        else:
            return [(name,self.get_block_position_from_mj_id(self.objects_mjdata_dict[name].id)) for name in self.object_names]

    def get_all_block_positions_by_body_name(self)-> Dict[str, np.ndarray]:
        # print('self.object_names in get_all_block_positions_by_body_name:', self.object_names)
        joint_ids = [mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in self.object_names]
        # print('joint_ids:', joint_ids)
        body_ids = [self._mj_model.jnt_bodyid[joint_id] for joint_id in joint_ids]
        # print('body_ids:', body_ids)

        body_names = [mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids]
        # print('body_names:', body_names)

        body_names = [re.sub(r'[^a-zA-Z0-9_-]', '', body_name) for body_name in body_names]
        # print('body_names after clean:', body_names)

        
        return {body_name: self.get_block_position_from_mj_id(self.objects_mjdata_dict[name].id) for body_name, name in zip(body_names, self.object_names)}

    def set_block_position(self, block_id, position):
        """
        Set the position of a block in the simulation.
        Args:
            block_id: the id of the block to set the position of.
            position: the position to set the block to, position will be in format [x, y ,z].
        """
        joint_name = f"block{block_id}_fj"
        joint_id = self._mj_model.joint(joint_name).id
        pos_adrr = self._mj_model.jnt_qposadr[joint_id]
        self._mj_data.qpos[pos_adrr:pos_adrr + 3] = position

    def set_all_block_positions(self, positions):
        """
        Set the positions of all blocks in the simulation.
        Args:
            positions: a list of positions to set the blocks to, positions will be in format [[x, y ,z], ...].
        """
        # set blocks positions
        for i, pos in enumerate(positions):
            self.set_block_position(i, pos)

    
    def set_all_items_positions(self, positions_list: List[tuple[str, list]]):
        """
        Set the positions of all items in the simulation.
        Args:
            positions: a list of tuples of item names and positions to set the items to, positions will be in format [(name, [x, y ,z]), ...].
        """
        # set items positions
        for name, pos in positions_list:
            # print('setting position for item:', name, pos)
            self.set_item_position(name, pos) 

    def set_item_position(self, item_name, position):
        joint_id = self._mj_model.joint(item_name).id
        pos_adrr = self._mj_model.jnt_qposadr[joint_id]
        self._mj_data.qpos[pos_adrr:pos_adrr + 3] = position
