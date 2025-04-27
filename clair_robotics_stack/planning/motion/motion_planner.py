import os
from klampt.math import se3, so3
from klampt import WorldModel, Geometry3D, RobotModel, vis
from klampt.model import collide
from klampt.model.geometry import box
from .abstract_motion_planner import AbstractMotionPlanner
import numpy as np


class MotionPlanner(AbstractMotionPlanner):
    def _get_klampt_world_path(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        world_path = os.path.join(dir, "klampt_world.xml")
        return world_path

    def add_attachments(self, robot, attachments=None, item_info=None):
        """
        Adds visual and geometric representations of specified attachments to the end-effector (robot link) of a robot model.

        Parameters:
        - robot: The robot object whose end-effector geometry is being updated to include the specified attachments and safety box.
        - attachments: A list of attachment names to be added. Valid options include "gripper", "camera", "spray-bottle", "red-can", "water-bottle", and "green-mug".
        - item_info: Optional dictionary containing information about the item being added. Should include dimensions, center, and color.

        Behavior:
        - Depending on the supplied `attachments`, creates various 3D geometric representations of the objects and aggregates them into a composite geometry group (`all_attachments_geom`).
        - Each attachment is defined as a rectangular box, with specific dimensions and center positions.
        - Updates the visual appearance color of certain attachments if `item_info` is provided, or assigns predefined colors for specific objects.
        - Adds a safety box geometry to the tool attachment area, regardless of the supplied attachments.
        - Transforms the composite geometry to consider the `ee_offset` parameter, ensuring proper alignment to the tool's frame.
        - Sets the final composite geometry to the "ee_link" of the robot, visually and geometrically representing all added attachments and the safety box.
        """
        if attachments is None:
            all_attachments = self.default_attachments[robot.name].copy()
        else:
            all_attachments = [*attachments]

        self._add_attachments(robot, all_attachments, item_info)

    def _add_attachments(self, robot, attachments, item_info=None):
        """
        add attachments to the robot. This is very abstract geometry that should be improved later.
        """
        # todo: the function need big changes.
        #      need to have an option to add only one item and not rebuild the whole thing every time.

        all_attachments_geom = Geometry3D()
        all_attachments_geom.setGroup()

        element = 0
        for attachment in attachments:
            if attachment == "gripper":
                gripper_obj = box(0.09, 0.09, 0.15, center=[0, 0, 0.07])
                gripper_geom = Geometry3D()
                gripper_geom.set(gripper_obj)
                all_attachments_geom.setElement(element, gripper_geom)
                element += 1

            elif attachment == "camera":
                camera_obj = box(0.18, 0.11, 0.06, center=[0, -0.05, 0.01])
                camera_geom = Geometry3D()
                camera_geom.set(camera_obj)
                all_attachments_geom.setElement(element, camera_geom)
                element += 1

            elif item_info and attachment is not None:
                item_dim = item_info.get("dimensions").get("conture_dimensions")
                item_center = item_info.get("dimensions").get("center")
                item_color = item_info.get("color").get("rgb")

                attachment_obj = box(*item_dim, center=item_center)
                attachment_geom = Geometry3D()
                attachment_geom.set(attachment_obj)
                all_attachments_geom.setElement(element, attachment_geom)
                element += 1

        # add safety box around where the tool cable is attached
        safety_box = box(0.13, 0.13, 0.03, center=[0, 0, -0.04])
        safety_box_geom = Geometry3D()
        safety_box_geom.set(safety_box)
        all_attachments_geom.setElement(element, safety_box_geom)

        # the positions of tools were measured for ee_offset = 0. move them back by ee_offset
        for i in range(all_attachments_geom.numElements()):
            element = all_attachments_geom.getElement(i)
            # x is forward in ff frame. nothing makes sense anymore...
            element.transform(so3.identity(), [0, 0, -self.ee_offset])
            all_attachments_geom.setElement(i, element)

        robot.link("ee_link").geometry().set(all_attachments_geom)

    def add_object_to_world(self, name, item):
        """
        Add a new object to the world.
        :param name: Name of the object.
        :param item: Dictionary containing the following keys:
            - geometry_file: Path to the object's geometry file.
            - coordinates: [x, y, z] coordinates.
            - angle: Rotation matrix (so3).
            - color: rgb array
            - scale: Scaling factor of the object (default is 1,1,1).
        """

        obj = self.world.makeRigidObject(name)
        geom = obj.geometry()
        if not geom.loadFile(item["geometry_file"]):
            raise ValueError(f"Failed to load geometry file: {item['geometry_file']}")

        # Set the transformation (rotation + position)
        if len(item["angle"]) != 9:
            item["angle"] = so3.rotation(item["angle"], np.pi / 2)
        transform = (item["angle"], item["coordinates"])
        geom.setCurrentTransform(*transform)
        if isinstance(item["scale"], float) or isinstance(item["scale"], int):
            geom.scale(item["scale"])
        else:
            geom.scale(*item["scale"])

        # Set the transformation for the rigid object
        obj.setTransform(*transform)

        # Set the object's color
        obj.appearance().setColor(*item["color"]["rgb"])

        # world collider need to be reinitialized after adding
        self.world_collider = collide.WorldCollider(self.world)

        # Save the object in the dictionary
        self.objects[name] = obj

        return obj

    def remove_object(self, name, vis_state=False):
        """
        Remove an object from the world and the dictionary.
        :param name: Name of the object to be removed.
        :param vis_state: Boolean to visualize the workspace after removing the object.
        """
        # if vis.shown():
        #     vis_state = True
        #     vis.show(False)
        #     time.sleep(0.3)
        self._remove_object(name)
        # if vis_state:
        #     self.visualize(window_name="workspace")

    def _remove_object(self, name):
        """
        Remove an object from the world and the dictionary.
        :param name: Name of the object to be removed.
        """
        obj = self.objects.pop(name, None)  # Remove from the dictionary
        if obj is None:
            print(f"Object '{name}' not found. Cannot remove.")
        else:
            self.world.remove(obj)
            print(f"Object '{name}' removed from the dictionary and world.")

    def remove_attachments(self, robot):
        """
        Removes attachments from the specified robot and resets them to default.

        Parameters:
        robot (Robot): The robot object whose attachments are being reset.

        This method first removes all current attachments of the given robot
        and then adds back the default attachments corresponding to
        the robot's name.
        """
        self._add_attachments(robot, *self.default_attachments[robot.name])
