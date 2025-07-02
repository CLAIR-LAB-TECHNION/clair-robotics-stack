import os

from klampt import Geometry3D
from klampt.model.geometry import box
from klampt import vis

from clair_robotics_stack.planning.motion.abstract_motion_planner import AbstractMotionPlanner
from .configurations import *


class SimulationMotionPlanner(AbstractMotionPlanner):
    def __init__(self):
        super().__init__(ee_offset=0.00, eps=0.05)
        self.ur5e_1 = self.world.robot("ur5e_1")
        self.ur5e_2 = self.world.robot("ur5e_2")

        self.ee_link = self.ur5e_2.link("ee_link")

    def _get_klampt_world_path(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        world_path = os.path.join(dir, "klampt_world.xml")
        return world_path

    def _add_attachments(self, robot, attachments):
        pass

    def attach_box_to_ee(self):
        """
        attach a box to the end effector for collision detection. Should be called once
        """
        # Note that the order is different here, width is in z direction
        sx, sy, sz = block_size
        box_obj = box(width=sz, height=sy, depth=sx, center=[0, 0, grasp_offset])
        box_geom = Geometry3D()
        box_geom.set(box_obj)

        self.ee_link.geometry().set(box_geom)

    def detach_box_from_ee(self):
        """
        detach the box from the end effector
        """
        dummy_box_obj = box(width=0.001, height=0.001, depth=0.001, center=[0, 0, 0])
        dummy_box_geom = Geometry3D()
        dummy_box_geom.set(dummy_box_obj)

        self.ee_link.geometry().set(dummy_box_geom)

    def move_block(self, name, position):
        """
        move block to position
        """
        rigid_obj = self.world.rigidObject(name)
        width, depth, height = block_size
        box_obj = box(width=width, height=height, depth=depth, center=position)
        rigid_obj.geometry().set(box_obj)

    def add_block(self, name, position, color=(0.3, 0.3, 0.3, 0.8)):
        """
        add block to the world
        """
        self._add_box_geom(name, block_size, position, color)

    def _add_box_geom(self, name, size, center, color, update_vis=True):
        """
        add box geometry for collision in the world
        """
        width, depth, height = size
        box_obj = box(width=width, height=height, depth=depth, center=center)
        box_geom = Geometry3D()
        box_geom.set(box_obj)
        box_rigid_obj = self.world.makeRigidObject(name)
        box_rigid_obj.geometry().set(box_geom)
        box_rigid_obj.appearance().setColor(*color)

        if update_vis:
            vis.add("world", self.world)
