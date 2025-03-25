from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from .entity import Entity
from ..episode.specs.robot_spec import RobotSpec
from ..common.defs.types import InfoDict
from ..episode.specs.camera_spec import CameraSpec, CameraType
from ..rendering import OffscreenRenderer

from ..common.transform import Transform
from ..common.ur5e_fk import forward

if TYPE_CHECKING:
    from .simulator import Simulator


class RobotAgent:
    """
    Represents an acting agent corresponding to a robot in the simulation.

    public fields:
        - spec:
            The robot specification object with which the robot model was loaded.
        - entity:
            An `Entity` object (see `spear_env.simulation.entity.Entity`) binding the robot model to the simulation
            physics.
        - sensor_map:
            A dictionary (str --> Sensor) where Sensor is a `spear_env.simulation.Entity` object representing a sensor
            in the model. The keys are the sensor type and the identifiers of the sensors in the model, delimited by two
            colon symbols. For example, the key for a force sensor whose identifier is "my_robot/sensor1" will be
            "force::my_robot/sensor1".
        - actuators:
            An entity `Entity` object (see `spear_env.simulation.entity.Entity`) binding the list of actuator elements
            in the model over which the agent has control.
    """

    def __init__(self, spec: RobotSpec, sim: Simulator, namespace: str):
        """
        Creates a new agent object for the simulation robot.
        :param spec: The robot specification object.
        :param sim: The simulation in which the robot resides.
        :param namespace: A unique namespace for the robot to avoid name conflicts.
        """
        self.spec = spec
        self.namespace = namespace
        self.entity = Entity(sim.composer.get_mounted_robot(self.spec), sim.physics)

        # Create sensor_map with namespaces
        self.sensor_map = {
            f'{sensor.element_tag}::{sensor.identifier}': sensor
            for sensor in Entity.from_list(sim.mjcf_model.find_all('sensor'), sim.physics)
            if self.namespace in sensor.identifier
        }

        # Create camera_map with namespaces
        self.camera_map = {}
        self.camera_spec = None
        for camera in Entity.from_list(sim.mjcf_model.find_all('camera'), sim.physics):
            if self.namespace in camera.identifier:
                camera_spec = CameraSpec(
                    identifier=camera.identifier,
                    height=480,  # Set default values or extract from camera attributes if available
                    width=640,
                    type=CameraType.RGBD  # Set default type or extract from camera attributes if available
                )
                name, renderer = self.__get_camera_name_and_renderer(camera_spec, sim)
                self.camera_map[name] = renderer
                self.camera_id = sim.model.camera(camera.identifier).id

        # Filter actuators based on namespace
        actuator_elements = [
            actuator for actuator in sim.composer.mjcf_model.find_all('actuator')
            if self.namespace in actuator.full_identifier
        ]
        self.actuators = Entity(actuator_elements, sim.physics)

        # Initialize MjvCamera
        self.mjv_camera = mujoco.MjvCamera()
        self.set_camera_parameters()

        # Extract the robot pose in the world frame
        self.robot_pose = self.get_robot_pose(sim)

    def get_robot_pose(self, sim):
        # Find the base body of the robot
        base_body = None
        bodies = sim.mjcf_model.find_all('body')
        for body in bodies:
            if hasattr(body, 'full_identifier'):
                if self.namespace in body.full_identifier:
                    robot_base_index = bodies.index(body)  # + 1
                    base_body = bodies[robot_base_index]
                    break

        if base_body is None:
            raise ValueError(f"Could not find base body for robot with namespace {self.namespace}")

        # Get the body ID
        body_id = sim.model.body(base_body.full_identifier).id

        # Get position and orientation
        position = sim.data.xpos[body_id]
        orientation = sim.data.xmat[body_id].reshape(3, 3)

        # Create and return a Transform object
        return Transform(rotation=orientation, translation=position, from_frame='world',
                         to_frame=f'{self.namespace}_base')

    def set_camera_parameters(self):
        self.mjv_camera.azimuth = 90
        self.mjv_camera.elevation = -45
        self.mjv_camera.distance = 2.0
        self.mjv_camera.lookat = np.array([0.0, 0.0, 0.0])

    def set_action(self, ctrl: npt.NDArray[np.float64]) -> None:
        """
        Set the control values for the actuators controlled by the agent.
        :param ctrl: The control values to set.
        """
        self.actuators.ctrl[:] = ctrl

    def reset(self):
        """Resets the joint positions of the robot."""
        self.entity.configure_joints(position=self.spec.init_pos, velocity=self.spec.init_vel)

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        The observation space of the agent as per the `gymnasium.spaces` standard. This includes joint positions,
        velocities, sensor data, and camera data.
        """
        pos_len = len(self.entity.get_joint_positions())
        vel_len = len(self.entity.get_joint_velocities())
        pos_bounds = self.entity.get_joint_ranges()
        vel_bounds = [[-np.inf, np.inf]] * vel_len
        state_lows, state_highs = np.concatenate([pos_bounds, vel_bounds]).T

        spaces = {
            "robot_state": gym.spaces.Box(
                low=state_lows,
                high=state_highs,
                shape=(pos_len + vel_len,),
                dtype=np.float64
            )
        }

        # Add sensor spaces
        spaces.update({
            'sensor': sensor.sensordata_space
            for _, sensor in self.sensor_map.items()
        })

        # Add camera spaces
        spaces.update({
            'camera': gym.spaces.Box(
                low=0,
                high=255,
                shape=self.render_camera(renderer).shape,
                dtype=np.float32
            )
            for _, renderer in self.camera_map.items()
        })

        spaces.update({
            'camera_pose': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, 7),
                dtype=np.float32
            )
        })

        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> gym.spaces.Space:
        """The action space of the agent as per the `gymnasium.spaces` standard."""
        bounds = self.actuators.ctrlrange.copy().astype(float)
        limits = self.actuators.ctrllimited.copy().astype(float)
        bounds[limits == 0] = [-np.inf, np.inf]
        low, high = bounds.T

        return gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def get_obs(self, sim: Simulator) -> dict[str, npt.NDArray[np.float64]]:
        """
        Retrieves the current agent observation.
        :return: A dictionary of agent observations.
        """
        out = dict(
            robot_state=np.concatenate([self.entity.get_joint_positions(), self.entity.get_joint_velocities()])
        )

        out.update({
            'sensor': sensor.sensordata
            for _, sensor in self.sensor_map.items()
        })

        # out.update({
        #     'camera': self.render_camera(renderer)
        #     for _, renderer in self.camera_map.items()
        # })
        #
        # pose_q = np.array(Transform(rotation=sim.data.cam_xmat[self.camera_id].reshape(3, 3),
        #                             translation=sim.data.cam_xpos[self.camera_id]).to_pose_quaternion().tolist())
        #
        # out.update({
        #     'camera_pose': pose_q
        # })

        out.update({
            'camera': None
        })

        out.update({
            'camera_pose': None
        })

        return out

    def get_info(self) -> InfoDict:
        """
        Retrieves an agent-specific information dictionary.
        :return: A dictionary of agent information.
        """
        return dict(
            qpos=self.entity.get_joint_positions(),
            qvel=self.entity.get_joint_velocities(),
        )

    @staticmethod
    def render_camera(renderer: OffscreenRenderer) -> np.ndarray:
        return renderer.render()

    @staticmethod
    def __get_camera_name_and_renderer(cam_spec: CameraSpec, sim: Simulator):
        identifier = cam_spec.identifier
        camera_id = sim.model.camera(identifier).id
        renderer = OffscreenRenderer(sim.model, sim.data, camera_id, width=cam_spec.width, height=cam_spec.height,
                                     depth=cam_spec.depth, segmentation=cam_spec.segmentation)
        dims = f'{renderer.height}X{renderer.width}'
        img_type = 'rgbd' if cam_spec.depth else 'rgb'
        name = f'camera_{identifier}_{dims}_{img_type}'

        return name, renderer
