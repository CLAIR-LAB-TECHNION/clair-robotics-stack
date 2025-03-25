from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
from .base_spec import Spec
from .robot_spec import RobotSpec
from .scene_spec import SceneSpec
from .task_spec import TaskSpec


@dataclass(frozen=True, eq=False)
class EpisodeSpec(Spec):
    """
    A specification for an episode to be run in the environment. It defines the simulated scene, the robot agents, and
    the tasks to be performed.

    public fields:
        - scene: A specification of the scene to be simulated
        - robots: A dictionary of robot specifications in the simulation.
        - tasks: A dictionary of task specifications for each robot.
    """
    scene: SceneSpec
    robots: Dict[str, RobotSpec]
    tasks: Dict[str, TaskSpec]

    def __post_init__(self):
        if not isinstance(self.scene, SceneSpec):
            super().__setattr__('scene', SceneSpec(self.scene))
        if not all(isinstance(robot, RobotSpec) for robot in self.robots.values()):
            super().__setattr__('robots', {name: RobotSpec(robot) for name, robot in self.robots.items()})
        if not all(isinstance(task, TaskSpec) for task in self.tasks.values()):
            super().__setattr__('tasks', {name: TaskSpec(task) for name, task in self.tasks.items()})

    @staticmethod
    def require_different_models(episode1: 'EpisodeSpec', episode2: 'EpisodeSpec') -> bool:
        return (SceneSpec.require_different_models(episode1.scene, episode2.scene) or
                any(RobotSpec.require_different_models(episode1.robots[name], episode2.robots[name])
                    for name in episode1.robots.keys() & episode2.robots.keys()))
