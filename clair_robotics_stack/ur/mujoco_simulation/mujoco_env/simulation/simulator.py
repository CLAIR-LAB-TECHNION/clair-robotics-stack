from typing import Optional, List, Dict
import numpy as np
from dm_control import mjcf
import mujoco
from .entity import Entity
from .mjcf_composer import MJCFComposer
from .mjcf_utils import physics_from_mjcf_model
from .robot_agent import RobotAgent
from ..common.defs.types import InfoDict, Vector
from ..episode.specs.robot_spec import RobotSpec
from ..episode.specs.scene_spec import SceneSpec


class Simulator:
    """
    An interface for initializing and interacting with MuJoCo simulations.
    A `Simulator` instance uses specifications for a scene and robots to compose a model for the simulation.
    Specifications are defined according to the spec dataclasses (see spear_env.episode.specs). A specification points
    to MJCF (XML) asset files that are loaded and merged according to the specification details.

    public fields:
    - scene / robots:
      The scene and robot specification instances set at instance construction or using the `swap_specs` method.
    - composer:
      An MJCF composition tool for merging episode-specific MJCF files.
    - physics:
      A representation of the simulation physics (see dm_control.mjcf.Physics). This is a wrapper around the
      MuJoCo model and data objects.
    - mjcf_model:
      An access variable to the MJCF model object on which the simulation model is based.
    - model:
      An access variable to the pointer of the MuJoCo model object in the simulation physics.
    - data:
      An access variable to the pointer of the MuJoCo data object in the simulation physics.

    public methods:
    - initialize:
      Composes and initializes the simulation model.
    - step:
      Steps the simulation a specified number of time ticks.
    - reset:
      Resets the simulation to the initial state.
    - swap_specs:
      Swaps the scene and robot specifications of the simulation.
    - get_agents:
      Creates new RobotAgent instances for the robots in the simulation.
    - get_privileged_info:
      Constructs an information dictionary containing privileged information.
    - free:
      Frees the simulation physics memory.
    - get_entity:
      Get an `Entity` object bound to an element in the simulation.
    """

    def __init__(self, scene: SceneSpec, robots: Dict[str, RobotSpec]) -> None:
        """
        Creates a new MuJoCo simulation according to the given scene and robot specifications.
        :param scene: the scene specification.
        :param robots: a dictionary of robot specifications.
        """
        # set scene and robot specifications
        self.scene = scene
        self.robots = robots

        # declare MJCF composition tool
        self.composer = MJCFComposer()

        # declare simulation keyframes
        self.keyframes: dict[str, tuple[mjcf.RootElement, Vector, Vector, Vector, Vector]] = {}

        # initialize simulation fields
        self.physics: Optional[mjcf.Physics] = None

        # initialize the simulation model and data
        self.initialize()

    def __del__(self) -> None:
        """Frees the simulation physics memory when the instance is deleted."""
        self.free()

    # ========================= #
    # ========== API ========== #
    # ========================= #

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """An access variable to the MJCF model object on which the simulation model is based."""
        return self.composer.mjcf_model

    @property
    def model(self) -> mujoco.MjModel:
        """An access variable to the pointer of the MuJoCo model object in the simulation physics."""
        return self.physics.model.ptr

    @property
    def data(self) -> mujoco.MjData:
        """An access variable to the pointer of the MuJoCo data object in the simulation physics."""
        return self.physics.data.ptr

    def initialize(self) -> None:
        """Initialize the MJCF models for the scene and each robot."""
        self.composer.set_base_scene(self.scene)
        for i, (robot_name, robot_spec) in enumerate(self.robots.items()):
            namespace = f"robot_{i}"
            self.composer.attach_robot(robot_spec, namespace)
        self.keyframes = self.composer.extract_keyframes()
        self.physics = physics_from_mjcf_model(self.composer.mjcf_model)

    def step(self, n_frames: int) -> None:
        """
        Step the simulator a specified number of time ticks
        :param n_frames: The number of simulated time ticks to step.
        """
        self.physics.step(nstep=n_frames)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def reset(self) -> None:
        """Resets the simulation to the initial state."""
        self.physics.reset()
        for agent in self.get_agents():
            agent.reset()
        self.__set_keyframe_state()
        self.__set_init_state()

    def swap_specs(self, scene: SceneSpec, robots: Dict[str, RobotSpec]) -> None:
        """
        Swaps the scene and robot specifications of the simulation to accommodate multiple robots.
        This method is used to change the scene and robots such that the simulation is reinitialized
        only when it is absolutely necessary.
        :param scene: the new scene specification.
        :param robots: a list of new robot specifications.
        """
        # Swap scene
        self.scene, scene = scene, self.scene

        # Determine if a new model is required by comparing each robot spec with its counterpart
        new_model_required = SceneSpec.require_different_models(self.scene, scene)
        if not new_model_required:
            for robot in self.robots.keys():
                old_robot, new_robot = self.robots[robot], robots[robot]
                # Check if different models are required for any robot pair
                if RobotSpec.require_different_models(old_robot, new_robot):
                    new_model_required = True
                    break

        # If new model is required, reinitialize the entire simulation
        if new_model_required:
            self.robots = robots  # Update the robots list to new specs
            self.initialize()
        else:
            # Otherwise, just update necessary parts
            for robot in self.robots.keys():
                old_robot, new_robot = self.robots[robot], robots[robot]
                self.composer.swap_spec_ids(self.scene.objects, scene.objects, [old_robot], [new_robot])
            # Update the robots reference to the new list after all swaps have been verified
            self.robots = robots

    def get_agents(self) -> List[RobotAgent]:
        """Create and return RobotAgent instances for each robot."""
        return [RobotAgent(robot, self, f"robot_{i}") for i, robot in enumerate(self.robots.values())]

    def get_privileged_info(self) -> InfoDict:
        """Returns a dictionary containing privileged information for each robot."""
        return {
            'model': self.model,
            'data': self.data,
            'robots': [{'robot_privileged_info': robot.spec.privileged_info} for robot in self.get_agents()]
        }

    def free(self) -> None:
        """Frees the simulation physics memory."""
        if self.physics is not None:
            self.physics.free()
            self.physics = None

    def get_entity(self, name: str, tag: str = 'body') -> Entity:
        """
        Get an `Entity` object (see `spear_env.simulation.entity.Entity`) bound to an element corresponding to the
        specified name and tag.
        :param name: the name attribute of the element to bind.
        :param tag: the XML tag of the element to bind.
        :return: An `Entity` object that binds the retrieved element.
        :raises: ValueError if the element is not found.
        """
        return Entity.from_name_and_tag(name, tag, self.mjcf_model, self.physics)

    # ================================== #
    # ========== init helpers ========== #
    # ================================== #

    def __compose_mjcf(self):
        self.composer.set_base_scene(self.scene)
        for i, (robot_name, robot_spec) in enumerate(self.robots.items()):
            namespace = f"robot_{i}"
            self.composer.attach_robot(robot_spec, namespace)
        self.keyframes = self.composer.extract_keyframes()

    def __set_keyframe_state(self):
        # get keyframe ID
        keyframe = self.scene.init_keyframe
        if keyframe is None:
            return  # no keyframe provided
        try:
            if isinstance(keyframe, str):
                keyframe_data = self.keyframes[keyframe]
            else:
                keyframe_data = list(self.keyframes.values())[keyframe]
        except (KeyError, IndexError):
            raise ValueError(f'Invalid keyframe ID: {keyframe}')
        for root, qpos, qvel, act, ctrl in keyframe_data:
            root_entity = Entity.from_model(root, self.physics)
            root_entity.set_state(qpos, qvel, act, ctrl, recursive=False)

    def __set_init_state(self):
        for robot in self.robots.values():
            # Initialize mount if applicable
            if robot.mount and hasattr(robot.mount, 'init_pos') and hasattr(robot.mount, 'init_vel'):
                # Directly access the mount element from the composer’s MJCF model (if mount is identifiable by name or tag)
                mount_entity = Entity(self.composer.mjcf_model.find('body', robot.mount.identifier), self.physics)
                mount_entity.configure_joints(robot.mount.init_pos, robot.mount.init_vel)
            # Initialize each attachment
            for attachment in robot.attachments:
                if hasattr(attachment, 'init_pos') and hasattr(attachment, 'init_vel'):
                    # Assuming attachments can be identified in a similar way
                    attachment_entity = Entity(self.composer.mjcf_model.find('body', attachment.identifier),
                                               self.physics)
                    attachment_entity.configure_joints(attachment.init_pos, attachment.init_vel)
