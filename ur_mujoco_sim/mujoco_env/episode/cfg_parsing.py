from __future__ import annotations
from typing import Union, TypeVar
from .specs import *
from ..common.defs.cfg_keys import *
from ..common.defs.types import Asset, Config

# represents the name of an internal asset or a configuration for such an asset
AssetOrConfig = Union[Asset, Config]

# the generic type that inherits from Spec
SpecT = TypeVar('SpecT', bound=Spec)


def episode_from_cfg(cfg: Config) -> EpisodeSpec:
    """
    Converts a configuration dictionary to an episode specification object.
    """
    scene = scene_spec_from_name_or_cfg(cfg[SCENE])
    robots = {name: robot_spec_from_name_or_cfg(robot_cfg) for name, robot_cfg in cfg[ROBOTS].items()}
    tasks = {name: task_spec_from_name_or_cfg(task_cfg) for name, task_cfg in cfg[TASKS].items()}
    return EpisodeSpec(scene, robots, tasks)


def scene_spec_from_name_or_cfg(asset_or_cfg: AssetOrConfig) -> SceneSpec:
    """
    Converts a configuration dictionary or resource name to a scene specification object.
    """
    if isinstance(asset_or_cfg, dict) and SCENE_OBJECTS in asset_or_cfg:
        asset_or_cfg[SCENE_OBJECTS] = __list_or_single_from_name_or_cfg(asset_or_cfg[SCENE_OBJECTS], ObjectSpec)
    return __type_from_name_or_cfg(asset_or_cfg, SceneSpec)


def robot_spec_from_name_or_cfg(asset_or_cfg: AssetOrConfig) -> RobotSpec:
    """
    Converts a configuration dictionary or resource name to a robot specification object.
    """
    if isinstance(asset_or_cfg, dict) and ROBOT_ATTACHMENTS in asset_or_cfg:
        asset_or_cfg[ROBOT_ATTACHMENTS] = __list_or_single_from_name_or_cfg(asset_or_cfg[ROBOT_ATTACHMENTS],
                                                                            AttachmentSpec)
    return __type_from_name_or_cfg(asset_or_cfg, RobotSpec)


def task_spec_from_name_or_cfg(asset_or_cfg: AssetOrConfig) -> TaskSpec:
    """
    Converts a configuration dictionary or an entrypoint string to a task specification object.
    """
    return __type_from_name_or_cfg(asset_or_cfg, TaskSpec)


def __list_or_single_from_name_or_cfg(asset_or_cfg: Union[AssetOrConfig, list[AssetOrConfig]],
                                      spec_type: type[SpecT]) -> list[SpecT]:
    if isinstance(asset_or_cfg, list):
        return [__type_from_name_or_cfg(data, spec_type) for data in asset_or_cfg]
    else:
        return [__type_from_name_or_cfg(asset_or_cfg, spec_type)]


def __type_from_name_or_cfg(inp: AssetOrConfig | SpecT, spec_cls: type[SpecT]) -> SpecT:
    if isinstance(inp, spec_cls):
        return inp
    if isinstance(inp, dict):
        if ADDON_BASE_JOINTS in inp:
            inp[ADDON_BASE_JOINTS] = __list_or_single_from_name_or_cfg(inp[ADDON_BASE_JOINTS], JointSpec)
        if JOINT_ACTUATORS in inp:
            inp[JOINT_ACTUATORS] = __list_or_single_from_name_or_cfg(inp[JOINT_ACTUATORS], ActuatorSpec)
        return spec_cls(**inp)
    else:
        return spec_cls(inp)
