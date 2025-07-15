""" a wrapper around spear env to simplify and fix some issues with the environment """
from copy import deepcopy, copy
from collections import namedtuple
import mujoco as mj
from clair_robotics_stack.ur.mujoco_simulation.mujoco_env import MujocoEnv
from clair_robotics_stack.ur.mujoco_simulation.mujoco_env.world_utils.object_manager import ObjectManager
from clair_robotics_stack.ur.mujoco_simulation.mujoco_env.world_utils.grasp_manager import GraspManager
from clair_robotics_stack.ur.mujoco_simulation.utils.logging_util import setup_logging
from clair_robotics_stack.ur.mujoco_simulation.mujoco_env.world_utils.configurations_and_constants import *


class SimEnv:
    def __init__(self, render_mode='rgb_array', cfg=muj_env_config, render_sleep_to_maintain_fps=True):
        print('start')
        self.render_mode = render_mode
        print('render_mode:', render_mode)
        self._env = MujocoEnv.from_cfg(cfg=cfg, render_mode=render_mode, frame_skip=frame_skip,
                                       sleep_to_maintain_fps=render_sleep_to_maintain_fps)
        print('self._env:', self._env)
        self.frame_skip = frame_skip
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['privileged']['model']
        self._mj_data = info['privileged']['data']
        self._env_entities = {name: agent.entity for name, agent in self._env.agents.items()}
        self.robots_joint_pos = {}
        self.robots_joint_velocities = {}
        self.robots_force = {}
        self.robots_camera = {}
        for agent in self._env_entities.keys():
            self.robots_joint_pos[agent] = np.zeros((1, 6))  # will be updated in reset
            self.robots_joint_velocities[agent] = np.zeros((1, 6))  # --""--
            self.robots_force[agent] = 0.0
            self.robots_camera[agent] = []
        self.gripper_state_closed = False  # --""--
        self.max_joint_velocities = INIT_MAX_VELOCITY

        self._object_manager = ObjectManager(self._mj_model, self._mj_data)
        self._grasp_manager = GraspManager(self._mj_model, self._mj_data, self._object_manager, min_grasp_distance=0.1)

        self.num_blocks = len(self._object_manager.object_names)

        self.image_res_h = 720
        self.image_res_w = 1280
        self.renderer = mj.Renderer(self._mj_model, self.image_res_h, self.image_res_w)
        self._mj_model.camera("robot-cam").fovy[0] = 45

        self._ee_mj_data = self._mj_data.body('robot_1_ur5e/robot_1_adhesive gripper/')
        # self.dt = self._mj_model.opt.timestep * frame_skip
        # self._pid_controller = PIDController(kp, ki, kd, dt)

        setup_logging()

    def close(self):
        self._env.close()

    def set_robot_joints(self, robot_name, joint_pos, joint_vel=(0,) * 6, simulate_step=True):
        self._env_entities[robot_name].set_state(position=joint_pos, velocity=joint_vel)
        if simulate_step:
            self.simulate_steps(1)

    def set_block_positions_on_table(self, block_positions_xy):
        z = 0.03
        self._object_manager.set_all_block_positions([[x, y, z] for x, y in block_positions_xy])

    def reset(self, randomize=True, block_positions=None):
        self.max_joint_velocities = INIT_MAX_VELOCITY

        obs, _ = self._env.reset()
        agents = obs.keys()

        for agent in agents:
            self.set_robot_joints(agent, [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, 0, 0], simulate_step=False)

        for agent in agents:
            self.robots_joint_pos[agent] = obs[agent]['robot_state'][:6]
            self.robots_joint_velocities[agent] = obs[agent]["robot_state"][6:12]
            # self.robots_force[agent] = obs[agent]['sensor']
            self.robots_camera[agent] = [obs[agent]['camera'], obs[agent]['camera_pose']]
        self.gripper_state_closed = False
        self._grasp_manager.release_object()
        print('block_positions: ', block_positions)
        self._object_manager.reset(randomize=randomize, block_positions=block_positions)

        self.step(self.robots_joint_pos, gripper_closed=False)

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def step(self, target_joint_pos, gripper_closed=None):
        # if reset_pid:
        #     self._pid_controller.reset_endpoint(target_joint_pos)
        if gripper_closed is None:
            gripper_closed = self.gripper_state_closed
        self.gripper_state_closed = gripper_closed

        self._env_step(target_joint_pos)
        self._clip_joint_velocities()

        if gripper_closed:
            if self._grasp_manager.attached_object_name is not None:
                self._grasp_manager.update_grasped_object_pose()
            else:
                self._grasp_manager.grasp_block_if_close_enough()
        else:
            self._grasp_manager.release_object()

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def simulate_steps(self, n_steps):
        """
        simulate n_steps in the environment without moving the robot
        """
        config = self.robots_joint_pos
        for _ in range(n_steps):
            self.step(config)

    def render(self):
        if self.render_mode == "human":
            return self._env.render()
        elif self.render_mode == "rgb_array":
            self._mj_data = self._env.sim.data
            self.renderer.update_scene(self._mj_data, "robot-cam")
            print("self.renderer._depth_rendering 128:",self.renderer._depth_rendering)
            rgb = self.renderer.render()
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()
            return rgb, depth
            # return rgb
        elif self.render_mode == "depth_array":
            self._mj_data = self._env.sim.data
            self.renderer.update_scene(self._mj_data, "robot-cam")
            self.renderer.enable_depth_rendering()
            print("self.renderer._depth_rendering 139:",self.renderer._depth_rendering)
            return self.renderer.render()
        return None

    def get_state(self):
        # object_positions = self._object_manager.get_all_block_positions_dict()
        state = {"robots_joint_pos": self.robots_joint_pos,
                 "robots_joint_velocities": self.robots_joint_velocities,
                 # "robots_force": self.robots_force,
                 # "robots_camera": self.robots_camera,
                 "gripper_state_closed": self.gripper_state_closed,
                 # "object_positions": object_positions,
                 "grasped_object": self._grasp_manager.attached_object_name,}
                 # "geom_contact": convert_mj_struct_to_namedtuple(self._env.sim.data.contact)}

        return deepcopy(state)

    def get_tower_height_at_point(self, point):
        block_positions = self._object_manager.get_all_block_positions_dict()
        not_grasped_block_positions = {name: pos for name, pos in block_positions.items()
                                       if name != self._grasp_manager.attached_object_name}
        not_grasped_block_positions = np.array(list(not_grasped_block_positions.values()))

        blocks_near_point = not_grasped_block_positions[
            np.linalg.norm(not_grasped_block_positions[:, :2] - point, axis=1) < 0.03]
        highest_block_height = np.max(blocks_near_point[:, 2]) if blocks_near_point.size > 0 \
            else not_grasped_block_positions[:, 2].min() - 0.02
        # if no blocks near point, return zero height which is estimated as lowest block minus block size
        return copy(highest_block_height)

    def get_block_positions(self):
        return list(self._object_manager.get_all_block_positions_dict().values())

    def set_gripper(self, closed: bool):
        """
        close/open gripper and don't change robot configuration
        @param closed: true if gripper should be closed, false otherwise
        @return: None
        """
        self.step(self.robots_joint_pos, closed)

    def _clip_joint_velocities(self):
        # new_vel = self.robots_joint_velocities.copy()
        # for agent, vel in new_vel.items():
        #     new_vel[agent] = np.clip(vel, -self.max_joint_velocities, self.max_joint_velocities)
        #     self._env_entities[agent].set_state(velocity=new_vel[agent])
        # self.robots_joint_velocities = new_vel
        return

    def _env_step(self, target_joint_pos):
        """ run environment step and update state of self accordingly"""

        # joint_control = self._pid_controller.control(self.robot_joint_pos)  # would be relevant if we change to force
        # control and use PID controller, but we are using position control right now.

        # gripper control for the environment, which is the last element in the control vector is completely
        # ignored right now, instead we attach the nearest graspable object to the end effector and maintain it
        # with the grasp manager, outside the scope of this method.

        # action = np.concatenate((target_joint_pos, [int(gripper_closed)])) # would have been if grasping worked
        actions = {}
        for agent, action in target_joint_pos.items():
            actions[agent] = np.concatenate((action, [0]))

        obs, r, term, trunc, info = self._env.step(actions)
        for agent, ob in obs.items():
            self.robots_joint_pos[agent] = ob['robot_state'][:6]
            self.robots_joint_velocities[agent] = ob['robot_state'][6:12]
            # self.robots_force[agent] = obs[agent]['sensor']
            self.robots_camera[agent] = [obs[agent]['camera'], obs[agent]['camera_pose']]

    def get_ee_pos(self):
        return deepcopy(self._ee_mj_data.xpos)

    def render_image_from_pose(self, position, rotation_matrix):

        cam = self._mj_data.camera("robot-cam")
        cam.xpos = position
        cam.xmat = rotation_matrix.T.flatten()

        self.renderer.update_scene(self._mj_data, "robot-cam")

        return self.renderer.render()

    def get_robot_cam_intrinsic_matrix(self):
        cam_model = self._mj_model.camera("robot-cam")
        fovy = cam_model.fovy[0]
        res_x = self.image_res_w
        res_y = self.image_res_h

        # Convert fovy from degrees to radians
        fovy_rad = np.deg2rad(fovy)

        # Calculate focal length
        f = res_y / (2 * np.tan(fovy_rad / 2))

        # Calculate principal point
        cx = res_x / 2
        cy = res_y / 2

        # Create intrinsic matrix
        intrinsic_matrix = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])


        return intrinsic_matrix

    def is_object_grasped(self):
        return self._grasp_manager.attached_object_name is not None

    def get_agent_joint(self, agent_name):
        return self.robots_joint_pos[agent_name]


def convert_mj_struct_to_namedtuple(mj_struct):
    """
    convert a mujoco struct to a dictionary
    """
    attrs = [attr for attr in dir(mj_struct) if not attr.startswith('__') and not callable(getattr(mj_struct, attr))]
    return namedtuple(mj_struct.__class__.__name__, attrs)(**{attr: getattr(mj_struct, attr) for attr in attrs})

