import sys
import time
from abc import abstractmethod

from frozendict import frozendict
import numpy as np
from numpy import pi
from klampt import WorldModel, Geometry3D, RobotModel
from klampt.model.geometry import box
from klampt import vis
from klampt.plan.cspace import MotionPlan
from klampt.plan import robotplanning
from klampt.model import ik
from klampt.math import se3, so3
from klampt.model import collide
import os


class AbstractMotionPlanner:
    default_attachments = frozendict(ur5e_1=["camera", "gripper"], ur5e_2=["gripper"])
    default_settings = frozendict({  # "type": "lazyrrg*",
        "type": "rrt*",
        "bidirectional": False,
        "connectionThreshold": 30.0,
        "perturbationRadius": 1.,
        # "suboptimalityFactor": 1.01,  # only for rrt* and prm*.
        # Don't use suboptimalityFactor as it's unclear how that parameter works...
        # seems like it's ignored even in rrt*
        # "shortcut": True, # only for rrt
    })
    # Class-level attribute to track initialization
    vis_initialized = False

    def __init__(self, eps=2e-2, attachments=default_attachments, settings=default_settings, ee_offset=0.15):
        """
        parameters:
        eps: epsilon gap for collision checking along the line in configuration space. Too high value may lead to
            collision, too low value may lead to slow planning. Default value is 1e-2.
        """
        self.eps = eps

        self.world = WorldModel()
        world_path = self._get_klampt_world_path()
        self.world.readFile(world_path)

        self.ee_offset = ee_offset
        self.ur5e_1 = self.world.robot("ur5e_1")
        self.ur5e_2 = self.world.robot("ur5e_2")
        self.robot_name_mapping = {"ur5e_1": self.ur5e_1, "ur5e_2": self.ur5e_2}
        for robot in self.robot_name_mapping.values():
            self._set_ee_offset(robot)
        self._add_attachments(self.ur5e_1, attachments["ur5e_1"])
        self._add_attachments(self.ur5e_2, attachments["ur5e_2"])

        self.world_collider = collide.WorldCollider(self.world)

        self.settings = frozendict(self.default_settings)

        self.objects = {}

    def is_pyqt5_available(self):
        try:
            import PyQt5
            return True
        except ImportError:
            return False

    def visualize(self, backend=None, window_name=None):
        """
        open visualization window
        """
        if AbstractMotionPlanner.vis_initialized:
            return

        if backend is None:
            if sys.platform.startswith('linux'):
                backend = "GLUT"
            else:
                backend = "PyQt5" if self.is_pyqt5_available() else "GLUT"

        vis.init(backend)
        if window_name:
            vis.createWindow(window_name)

        vis.add("world", self.world)
        # vis.setColor(('world', 'ur5e_1'), 0, 1, 1)
        # vis.setColor(('world', 'ur5e_2'), 0, 0, 0.5)

        # set camera position:
        viewport = vis.getViewport()
        viewport.camera.tgt = [0, -0.6, 0.5]
        viewport.camera.rot = [0, -0.4, 0.35]
        viewport.camera.dist = 3.3

        vis.show()
        AbstractMotionPlanner.vis_initialized = True

    def vis_config(self, robot_name, config_, vis_name="robot_config", rgba=(0, 0, 1, 0.5)):
        """
        Show visualization of the robot in a config
        :param robot_name:
        :param config_:
        :param rgba: color and transparency
        :return:
        """
        config = config_.copy()
        if len(config) == 6:
            config = self.config6d_to_klampt(config)
        config = [config]  # There's a bug in visualize config so we just visualize a path of length 1

        vis.add(vis_name, config)
        vis.setColor(vis_name, *rgba)
        vis.setAttribute(vis_name, "robot", robot_name)

    def vis_path(self, robot_name, path_):
        """
        show the path in the visualization
        """
        path = path_.copy()
        if len(path[0]) == 6:
            path = [self.config6d_to_klampt(q) for q in path]

        robot = self.robot_name_mapping[robot_name]
        robot.setConfig(path[0])
        robot_id = robot.id

        # trajectory = RobotTrajectory(robot, range(len(path)), path)
        vis.add("path", path)
        vis.setColor("path", 1, 1, 1, 0.5)
        vis.setAttribute("path", "robot", robot_name)

    def show_point_vis(self, point, name="point", rgba=(0, 1, 0, 0.5)):
        vis.add(name, point)
        vis.setColor(name, *rgba)

    def show_ee_poses_vis(self):
        """
        show the end effector poses of all robots in the
        """
        for robot in self.robot_name_mapping.values():
            ee_transform = robot.link("ee_link").getTransform()
            vis.add(f"ee_pose_{robot.getName()}", ee_transform)

    def update_robot_config(self, robot_name, config):
        if len(config) == 6:
            config = self.config6d_to_klampt(config)
        robot = self.robot_name_mapping[robot_name]
        robot.setConfig(config)

    def plan_from_start_to_goal_config(self, robot_name: str, start_config, goal_config, max_time=15,
                                       max_length_to_distance_ratio=10):
        """
        plan from a start and a goal that are given in 6d configuration space
        """
        start_config_klampt = self.config6d_to_klampt(start_config)
        goal_config_klampt = self.config6d_to_klampt(goal_config)

        robot = self.robot_name_mapping[robot_name]
        path = self._plan_from_start_to_goal_config_klampt(robot, start_config_klampt, goal_config_klampt,
                                                           max_time, max_length_to_distance_ratio)

        return self.path_klampt_to_config6d(path)

    def _plan_from_start_to_goal_config_klampt(self, robot, start_config, goal_config, max_time=15,
                                               max_length_to_distance_ratio=10):
        """
        plan from a start and a goal that are given in klampt 8d configuration space
        """
        robot.setConfig(start_config)

        planner = robotplanning.plan_to_config(self.world, robot, goal_config,
                                               # ignore_collisions=[('keep_out_from_ur3_zone', 'table2')],
                                               # extraConstraints=
                                               **self.settings)
        planner.space.eps = self.eps

        # before planning, check if a direct path is possible, then no need to plan
        if self._is_direct_path_possible(planner, start_config, goal_config):
            return [goal_config]

        return self._plan(planner, max_time, max_length_to_distance_ratio=max_length_to_distance_ratio)

    def _plan(self, planner: MotionPlan, max_time=15, steps_per_iter=1000, max_length_to_distance_ratio=10):
        """
        find path given a prepared planner, with endpoints already set
        @param planner: MotionPlan object, endpoints already set
        @param max_time: maximum planning time
        @param steps_per_iter: steps per iteration
        @param max_length_to_distance_ratio: maximum length of the pass to distance between start and goal. If there is
            still time, the planner will continue to plan until this ratio is reached. This is to avoid long paths
            where the robot just moves around because non-optimal paths are still possible.
        """
        start_time = time.time()
        path = None
        print("planning motion...")
        while (path is None or self.compute_path_length_to_distance_ratio(path) > max_length_to_distance_ratio) \
                and time.time() - start_time < max_time:
            planner.planMore(steps_per_iter)
            path = planner.getPath()
        print("planning took ", time.time() - start_time, " seconds.")
        if path is None:
            print("no path found")
        return path

    def plan_multiple_robots(self):
        # implement if\when necessary.
        # robotplanning.plan_to_config supports list of robots and goal configs
        raise NotImplementedError

    @staticmethod
    def config6d_to_klampt(config):
        """
        There are 8 links in our rob for klampt, some are stationary, actual joints are 1:7
        """
        config_klampt = [0] * 8
        config_klampt[1:7] = config
        return config_klampt

    @staticmethod
    def klampt_to_config6d(config_klampt):
        """
        There are 8 links in our rob for klampt, some are stationary, actual joints are 1:7
        """
        if config_klampt is None:
            return None
        return config_klampt[1:7]

    def path_klampt_to_config6d(self, path_klampt):
        """
        convert a path in klampt 8d configuration space to 6d configuration space
        """
        if path_klampt is None:
            return None
        path = []
        for q in path_klampt:
            path.append(self.klampt_to_config6d(q))
        return path

    def compute_path_length(self, path):
        """
        compute the length of the path
        """
        if path is None:
            return np.inf
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        return length

    def compute_path_length_to_distance_ratio(self, path):
        """ compute the ratio of path length to the distance between start and goal """
        if path is None:
            return np.inf
        start = np.array(path[0])
        goal = np.array(path[-1])
        distance = np.linalg.norm(start - goal)
        length = self.compute_path_length(path)
        return length / distance

    @abstractmethod
    def _add_attachments(self, robot, attachments):
        pass

    def _is_direct_path_possible(self, planner, start_config_, goal_config_):
        # EmbeddedRobotCspace only works with the active joints:
        start_config = self.klampt_to_config6d(start_config_)
        goal_config = self.klampt_to_config6d(goal_config_)
        return planner.space.isVisible(start_config, goal_config)

    def is_config_feasible(self, robot_name, config):
        """
        check if the config is feasible (not within collision)
        """
        if len(config) == 6:
            config_klampt = self.config6d_to_klampt(config)
        else:
            config_klampt = config.copy()

        if len(config) == 0:
            return False

        robot = self.robot_name_mapping[robot_name]
        current_config = robot.getConfig()
        robot.setConfig(config_klampt)

        # we have to get all collisions since there is no method for robot-robot collisions-+--
        all_collisions = list(self.world_collider.collisions())

        robot.setConfig(current_config)  # return to original motion planner state

        # all collisions is a list of pairs of colliding geometries. Filter only those that contains a name that
        # ends with "link" and belongs to the robot, and it's not the base link that always collides with the table.
        for g1, g2 in all_collisions:
            if g1.getName().endswith("link") and g1.getName() != "base_link" and g1.robot().getName() == robot_name:
                return False
            if g2.getName().endswith("link") and g2.getName() != "base_link" and g2.robot().getName() == robot_name:
                return False

        return True

    def get_forward_kinematics(self, robot_name, config):
        """
        get the forward kinematics of the robot, this already returns the transform to world!
        """
        if len(config) == 6:
            config_klampt = self.config6d_to_klampt(config)
        else:
            config_klampt = config.copy()

        robot = self.robot_name_mapping[robot_name]

        previous_config = robot.getConfig()
        robot.setConfig(config_klampt)
        link = robot.link("ee_link")
        ee_transform = link.getTransform()
        robot.setConfig(previous_config)

        return ee_transform

    def _set_ee_offset(self, robot):
        ee_transform = robot.link("ee_link").getParentTransform()
        ee_transform = se3.mul(ee_transform, (so3.identity(), (0, 0, self.ee_offset)))
        robot.link("ee_link").setParentTransform(*ee_transform)
        # reset the robot config to update:
        robot.setConfig(robot.getConfig())

    def ik_solve(self, robot_name, ee_transform, start_config=None):

        if start_config is not None and len(start_config) == 6:
            start_config = self.config6d_to_klampt(start_config)

        robot = self.robot_name_mapping[robot_name]
        return self.klampt_to_config6d(self._ik_solve_klampt(robot, ee_transform, start_config))

    def _ik_solve_klampt(self, robot, ee_transform, start_config=None):

        curr_config = robot.getConfig()
        if start_config is not None:
            robot.setConfig(start_config)

        ik_objective = ik.objective(robot.link("ee_link"), R=ee_transform[0], t=ee_transform[1])
        res = ik.solve(ik_objective, tol=2e-3, iters=10000)
        if not res:
            # print("ik not solved")
            robot.setConfig(curr_config)
            return None

        res_config = robot.getConfig()

        robot.setConfig(curr_config)

        return res_config

    @abstractmethod
    def _get_klampt_world_path(self):
        pass

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
        obj.appearance().setColor(*item["color"])

        # world collider need to be reinitialized after adding
        self.world_collider = collide.WorldCollider(self.world)

        # Save the object in the dictionary
        self.objects[name] = obj

        return obj

    def get_object(self, name):
        """
        Retrieve an Rigidobject object by name from the dictionary.
        :param name: Name of the object.
        :return: The object if found, otherwise None.
        """
        obj = self.objects.get(name)
        if obj is None:
            print(f"Object '{name}' not found.")
        return obj

    def animate_path(self, robot_name, path, distance_between_configs=0.1, sleep_between_configs=0.02,
                     sleep_between_waypoints=0.1):
        robot = self.robot_name_mapping[robot_name]
        robot.setConfig(self.config6d_to_klampt(path[0]))

        path = np.asarray(path)

        for c1, c2 in zip(path[:-1], path[1:]):
            diff = np.array(c2) - np.array(c1)
            distance = np.linalg.norm(diff)
            n_steps = int(distance / distance_between_configs)
            for i in range(1, n_steps + 1):
                config = c1 + (c2 - c1) * i / n_steps
                robot.setConfig(self.config6d_to_klampt(config))
                time.sleep(sleep_between_configs)
            time.sleep(sleep_between_waypoints)

    # def remove_object(self, name, vis_state=False):
    #     """
    #     Remove an object from the world and the dictionary.
    #     :param name: Name of the object to be removed.
    #     :param vis_state: Boolean to visualize the workspace after removing the object.
    #     """
    #     if vis.shown():
    #         vis_state = True
    #         vis.show(False)
    #         time.sleep(0.3)
    #     self._remove_object(name)
    #     if vis_state:
    #         self.visualize(window_name="workspace")
    #
    # def _remove_object(self, name):
    #     """
    #     Remove an object from the world and the dictionary.
    #     :param name: Name of the object to be removed.
    #     """
    #     obj = self.objects.pop(name, None)  # Remove from the dictionary
    #     if obj is None:
    #         print(f"Object '{name}' not found. Cannot remove.")
    #     else:
    #         self.world.remove(obj)
    #         print(f"Object '{name}' removed from the dictionary and world.")
    #


