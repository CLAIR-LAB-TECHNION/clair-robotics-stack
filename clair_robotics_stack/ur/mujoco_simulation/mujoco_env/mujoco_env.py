from __future__ import annotations
from typing import Optional, SupportsFloat, Any, Literal, Dict
import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType
from gymnasium.utils import EzPickle
from .common.defs.types import InfoDict, Config, FilePath
from .episode.samplers import EpisodeSampler, CfgEpisodeSampler, CfgFileEpisodeSampler
from .episode.specs.episode_spec import EpisodeSpec
from .simulation.robot_agent import RobotAgent
from .simulation.simulator import Simulator
from .tasks.task import Task
from .rendering import BaseRenderer, WindowRenderer, OffscreenRenderer


class MujocoEnv(Env, EzPickle):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array', 'segmentation'],
    }

    def __init__(
            self,
            episode_sampler: EpisodeSampler,
            render_mode: Optional[Literal['human', 'rgb_array', 'segmentation', 'depth_array']] = None,
            frame_skip: int = 1,
            reset_on_init: bool = True,
            sleep_to_maintain_fps: bool = True
    ) -> None:
        self.sleep_to_maintain_fps = sleep_to_maintain_fps

        EzPickle.__init__(self, episode_sampler, frame_skip, render_mode, reset_on_init)
        self.episode_sampler = episode_sampler
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.episode: Optional[EpisodeSpec] = None
        self.sim: Optional[Simulator] = None
        self.agents: Optional[Dict[str, RobotAgent]] = None
        self.tasks: Optional[Dict[str, Task]] = None
        self.renderer: Optional[BaseRenderer] = None
        self.observation_space = spaces.Box(np.inf, -np.inf, (0,))
        self.action_space = spaces.Box(np.inf, -np.inf, (0,))
        if reset_on_init:
            self.reset()

    @classmethod
    def from_cfg(
            cls,
            cfg: Config,
            render_mode: Optional[Literal['human', 'rgb_array', 'segmentation', 'depth_array']] = None,
            frame_skip: int = 1,
            reset_on_init: bool = True,
            sleep_to_maintain_fps: bool = True
    ) -> MujocoEnv:
        return cls(CfgEpisodeSampler(cfg), render_mode, frame_skip, reset_on_init,
                   sleep_to_maintain_fps=sleep_to_maintain_fps)

    @classmethod
    def from_cfg_file(
            cls,
            cfg_file: FilePath,
            render_mode: Optional[Literal['human', 'rgb_array', 'segmentation', 'depth_array']] = None,
            frame_skip: int = 1,
            reset_on_init: bool = True
    ) -> MujocoEnv:
        return cls(CfgFileEpisodeSampler(cfg_file), render_mode, frame_skip, reset_on_init)

    def step(self, action: Dict[str, ActType]) -> tuple[
        Dict[str, ObsType], Dict[str, SupportsFloat], Dict[str, bool], bool, dict[str, Any]]:
        for agent_name, task in self.tasks.items():
            task.begin_frame(action[agent_name])
        self.do_simulation(action, self.frame_skip)
        for agent_name, task in self.tasks.items():
            task.end_frame(action[agent_name])
        obs = {agent_name: agent.get_obs(self.sim) for agent_name, agent in self.agents.items()}
        rewards = {agent_name: task.score() for agent_name, task in self.tasks.items()}
        dones = {agent_name: task.is_done() for agent_name, task in self.tasks.items()}
        infos = {agent_name: agent.get_info() for agent_name, agent in self.agents.items()}
        return obs, rewards, dones, False, infos

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.__set_next_episode()
        with self.sim.physics.reset_context():
            self.sim.reset()
            for agent in self.agents.values():
                agent.reset()
            for agent_name, task in self.tasks.items():
                task.reset(**self.episode.tasks[agent_name].params)
        return {agent_name: agent.get_obs(self.sim) for agent_name, agent in self.agents.items()}, self.__get_info_dict()

    def render(self):
        if self.render_mode is None:
            raise AttributeError(f'Cannot render environment without setting `render_mode`. set to one of: '
                                 f'{self.metadata["render_modes"]}')
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.sim is not None:
            self.sim.free()
            self.sim = None

    @property
    def dt(self):
        return self.sim.physics.timestep() * self.frame_skip

    def set_episode(self, episode: EpisodeSpec):
        self.episode, episode = episode, self.episode
        if episode is None or self.sim is None or EpisodeSpec.require_different_models(self.episode, episode):
            self.initialize_simulation()
        else:
            self.sim.swap_specs(self.episode.scene, self.episode.robots)

        if episode is None or self.tasks is None or any(
                episode.tasks[agent_name].cls != self.episode.tasks[agent_name].cls for agent_name in
                self.episode.robots.keys()):
            self.tasks = {agent_name: self.episode.tasks[agent_name].cls(self.sim) for agent_name in
                          self.episode.robots.keys()}

        self.agents = {f'ur5e_{agent_name+1}': agent for agent_name, agent in enumerate(self.sim.get_agents())}
        self.observation_space = spaces.Dict(
            {agent_name: agent.observation_space for agent_name, agent in self.agents.items()})
        self.action_space = spaces.Dict({agent_name: agent.action_space for agent_name, agent in self.agents.items()})

    def initialize_simulation(self):
        if self.sim is None:
            self.sim = Simulator(self.episode.scene, self.episode.robots)
        else:
            self.sim.swap_specs(self.episode.scene, self.episode.robots)
        self.__initialize_renderer()

    def do_simulation(self, ctrl: Dict[str, ActType], n_frames: int):
        for agent_name, action in ctrl.items():
            # if np.array(action).shape != self.action_space.shape:
            #     raise ValueError(
            #         f"Action dimension mismatch for {agent_name}. Expected {self.action_space.shape}, found {np.array(action).shape}")
            self.agents[agent_name].set_action(action)
        self.sim.step(n_frames)

    def __set_next_episode(self):
        new_episode = self.episode_sampler.sample()
        self.set_episode(new_episode)

    def __initialize_renderer(self):
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        if self.renderer is not None:
            self.renderer.close()
        if self.render_mode == 'human':
            self.renderer = WindowRenderer(self.sim.model, self.sim.data, self.episode.scene.render_camera,
                                           render_fps=self.metadata["render_fps"],
                                           sleep_to_maintain_fps=self.sleep_to_maintain_fps,
                                           **self.episode.scene.renderer_cfg)
        else:
            self.renderer = OffscreenRenderer(self.sim.model, self.sim.data, self.episode.scene.render_camera,
                                              depth=self.render_mode == 'depth_array',
                                              segmentation=self.render_mode == 'segmentation',
                                              **self.episode.scene.renderer_cfg)

    def __get_info_dict(self) -> InfoDict:
        return dict(
            task={agent_name: task.get_info() for agent_name, task in self.tasks.items()},
            agents={agent_name: agent.get_info() for agent_name, agent in self.agents.items()},
            privileged=self.sim.get_privileged_info() if self.episode.robots[
                list(self.episode.robots.keys())[0]].privileged_info else {}
        )
