import numpy as np
from mujoco_simulation.motion_planning.simulation_motion_planner import SimulationMotionPlanner
from mujoco_simulation.mujoco_env.sim_env import SimEnv
import logging

FACING_DOWN_R = [[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]]


def compose_transformation_matrix(rotation, translation):
    # Set the upper-left 3x3 part to the rotation matrix
    rotation_flattened = np.array(rotation).flatten()

    # Set the upper-right 3x1 part to the translation vector
    translation = np.array(translation)

    return rotation_flattened, translation


def point_in_square(square_center, edge_length, point):
    # Calculate half the edge length to determine the bounds
    half_edge = edge_length / 2

    # Calculate the left, right, bottom, and top boundaries of the square
    left_bound = square_center[0] - half_edge
    right_bound = square_center[0] + half_edge
    bottom_bound = square_center[1] - half_edge
    top_bound = square_center[1] + half_edge

    # Check if the point is within these boundaries
    return (left_bound <= point[0] <= right_bound) and (bottom_bound <= point[1] <= top_bound)


def canonize_config(config, boundries=(1.2 * np.pi, np.pi, np.pi, 1.5 * np.pi, 1.5 * np.pi, 1.2 * np.pi) * 6):
    for i in range(6):
        while config[i] > boundries[i]:
            config[i] -= 2 * np.pi
        while config[i] < -boundries[i]:
            config[i] += 2 * np.pi
    return config


class MotionExecutor:
    def __init__(self, env: SimEnv):
        self.env = env
        self.motion_planner = SimulationMotionPlanner()
        self.env.reset()

        self.default_config = [0.0, -1.2, 0.8419, -1.3752, -1.5739, -2.3080]

        state = self.env.get_state()

        self.update_all_robots_configs_in_mp()

        self.time_step = self.env._mj_model.opt.timestep * self.env.frame_skip

    def update_all_robots_configs_in_mp(self):
        state = self.env.get_state()
        for robot, pose in state['robots_joint_pos'].items():
            self.motion_planner.update_robot_config(robot, pose)

    def moveJ(self, robot_name, target_joints, speed=1.0, acceleration=1.0, tolerance=0.003):
        self.zero_all_robots_vels_except(robot_name)

        current_joints = self.env.robots_joint_pos[robot_name]
        current_velocities = self.env.robots_joint_velocities[robot_name]

        logging.info(f"MovingJ {robot_name} from {current_joints} to {target_joints}")

        # Calculate the joint differences
        joint_diffs = target_joints - current_joints

        # Calculate the time needed for the movement based on max velocity and acceleration
        max_joint_diff = np.max(np.abs(joint_diffs))
        time_to_max_velocity = speed / acceleration
        distance_at_max_velocity = max_joint_diff - 0.5 * acceleration * time_to_max_velocity ** 2

        if distance_at_max_velocity > 0:
            total_time = 2 * time_to_max_velocity + distance_at_max_velocity / speed
        else:
            total_time = 2 * np.sqrt(max_joint_diff / acceleration)

        # Calculate the number of steps based on the frame skip and simulation timestep
        num_steps = int(total_time / self.time_step)
        num_steps = max(num_steps, 1)
        max_steps = num_steps * 2  # Set a maximum number of steps (2x the expected number)

        # Generate smooth joint trajectories
        trajectories = []
        for i, diff in enumerate(joint_diffs):
            if abs(diff) > tolerance/4:
                trajectory = self.generate_smooth_trajectory(current_joints[i], target_joints[i], num_steps)
                trajectories.append(trajectory)
            else:
                trajectories.append(np.full(num_steps, current_joints[i]))
        #TODO save orig other robot config
        for robot in self.env.robots_joint_pos.keys():
            if robot != robot_name:
                other_robot = robot

        other_robot_positions = self.env.get_agent_joint(other_robot)

        # Execute the trajectory
        for step in range(max_steps):
            if step < num_steps:
                target_positions = [traj[step] for traj in trajectories]
            else:
                target_positions = target_joints

            actions = {robot: self.env.robots_joint_pos[robot] for robot in self.env.robots_joint_pos.keys() if
                       robot != robot_name}
            actions[robot_name] = target_positions
            self.env.step(actions)
            self.env.set_robot_joints(robot_name=other_robot, joint_pos=other_robot_positions, simulate_step=False)

            # Check if we've reached the target joints
            current_joints = self.env.robots_joint_pos[robot_name]
            if np.allclose(current_joints, target_joints, atol=tolerance):
                break

        self.update_all_robots_configs_in_mp()

        if step == max_steps - 1:
            logging.warning(f"Movement for {robot_name} timed out before reaching the target joints.")


    def generate_smooth_trajectory(self, start, end, num_steps):
        t = np.linspace(0, 1, num_steps)
        trajectory = start + (end - start) * (3 * t ** 2 - 2 * t ** 3)
        return trajectory

    def moveJ_path(self, robot_name, path_configs, speed=1.0, acceleration=1.0, blend_radius=0.05, tolerance=0.003):
        """
        Move the robot through a path of joint configurations with blending.

        :param robot_name: Name of the robot to move
        :param path_configs: List of joint configurations to move through
        :param speed: Maximum joint speed
        :param acceleration: Maximum joint acceleration
        :param blend_radius: Blend radius for smoothing between configurations
        :param tolerance: Tolerance for considering a point reached
        """
        if len(path_configs) == 1:
            return self.moveJ(robot_name, path_configs[0], speed, acceleration, tolerance)

        self.zero_all_robots_vels_except(robot_name)

        logging.info(f"MovingJ path {robot_name} {path_configs}")

        path_configs = np.asarray(path_configs)
        full_trajectory = []

        for i in range(len(path_configs) - 1):
            start_config = np.array(path_configs[i])
            end_config = np.array(path_configs[i + 1])

            if i == 0:
                # For the first segment, start from the initial configuration
                segment_trajectory = self.generate_trajectory(start_config, end_config, speed, acceleration,
                                                              blend_radius, blend_start=True,
                                                              blend_end=(i < len(path_configs) - 2))
            elif i == len(path_configs) - 2:
                # For the last segment, end at the final configuration
                segment_trajectory = self.generate_trajectory(start_config, end_config, speed, acceleration,
                                                              blend_radius, blend_start=True, blend_end=False)
            else:
                # For middle segments, blend at both ends
                segment_trajectory = self.generate_trajectory(start_config, end_config, speed, acceleration,
                                                              blend_radius, blend_start=True, blend_end=True)

            full_trajectory.extend(segment_trajectory)

        # Execute the full trajectory
        self.execute_trajectory(robot_name, full_trajectory, tolerance)

        self.update_all_robots_configs_in_mp()

    def generate_trajectory(self, start_config, end_config, speed, acceleration, blend_radius, blend_start=True,
                            blend_end=True):
        distance = np.linalg.norm(end_config - start_config)
        total_time = distance / speed
        num_steps = max(int(total_time / self.time_step), 2)  # Ensure at least 2 steps

        trajectory = []

        for t in np.linspace(0, 1, num_steps):
            if blend_start and t < 0.5:
                # Apply blending at the start
                t_blend = t * 2
                config = start_config + (end_config - start_config) * blend_radius * (t_blend ** 2 * (3 - 2 * t_blend))
            elif blend_end and t >= 0.5:
                # Apply blending at the end
                t_blend = (t - 0.5) * 2
                config = start_config + (end_config - start_config) * (
                        1 - blend_radius * ((1 - t_blend) ** 2 * (3 - 2 * (1 - t_blend))))
            else:
                # Linear interpolation in the middle
                config = start_config + (end_config - start_config) * t

            trajectory.append(config)

        return trajectory

    def execute_trajectory(self, robot_name, trajectory, tolerance):
        max_steps = len(trajectory) * 2  # Set a maximum number of steps (2x the expected number)

        for step in range(max_steps):
            if step < len(trajectory):
                target_positions = trajectory[step]
            else:
                target_positions = trajectory[-1]

            actions = {robot: self.env.robots_joint_pos[robot] for robot in self.env.robots_joint_pos if
                       robot != robot_name}
            actions[robot_name] = target_positions
            self.env.step(actions)

            # Check if we've reached the final configuration
            current_joints = self.env.robots_joint_pos[robot_name]
            if np.allclose(current_joints, trajectory[-1], atol=tolerance):
                break

        if step == max_steps - 1:
            print(f"WARNING: Movement for {robot_name} timed out before reaching the final configuration.")

    def plan_and_moveJ(self, robot_name, target_joints, speed=1.0, acceleration=1.0, blend_radius=0.05, tolerance=0.003,
                       max_planning_time=5, max_length_to_distance_ratio=2, replan_from_other_if_not_found=True):
        curr_joint_state = self.env.robots_joint_pos[robot_name]

        logging.info(f"Planning and movingJ {robot_name} from {curr_joint_state} to {target_joints}")

        path = self.motion_planner.plan_from_start_to_goal_config(robot_name, curr_joint_state, target_joints,
                                                                  max_time=max_planning_time,
                                                                  max_length_to_distance_ratio=max_length_to_distance_ratio)
        if path is None:
            # retry from another start config
            print(f"coudn't find path from {curr_joint_state} to {target_joints}"
                  "trying from another initial config")

            logging.info("couldn't find plan.")

            if not replan_from_other_if_not_found:
                return False
            else:
                start_config = [0.2, -1.2, 1.7, -2., -1.5, 0]
                path_to_start = self.motion_planner.plan_from_start_to_goal_config(robot_name, start_config,
                                                                                   target_joints,
                                                                                   max_time=max_planning_time,
                                                                                   max_length_to_distance_ratio=max_length_to_distance_ratio)
                path_from_start = self.motion_planner.plan_from_start_to_goal_config(robot_name, start_config,
                                                                                     target_joints,
                                                                                     max_time=max_planning_time,
                                                                                     max_length_to_distance_ratio=max_length_to_distance_ratio)
                if path_to_start is None or path_from_start is None:
                    print(f"couldn't plan through intermediate config as well...")
                    logging.info("couldn't plan through intermediate config as well...")
                    return False
                path = path_to_start + path_from_start

        self.moveJ_path(robot_name, path, speed, acceleration, blend_radius=blend_radius, tolerance=tolerance)
        return True

    def plan_and_move_to_xyz_facing_down(self, robot_name, target_xyz, speed=1.0, acceleration=1.0, blend_radius=0.05,
                                         tolerance=0.003, max_planning_time=5, max_length_to_distance_ratio=2,
                                         cannonized_config=True):
        target_transform = compose_transformation_matrix(FACING_DOWN_R, target_xyz)
        goal_config = self.facing_down_ik(robot_name, target_transform, max_tries=50)

        logging.info(f"Planning and moving to xyz facing down {robot_name} from"
                     f" {self.env.robots_joint_pos[robot_name]} to {goal_config}"
                     f" target: {target_xyz}")

        if goal_config is None or len(goal_config) == 0:
            print(f"WARNING: IK solution not found for {robot_name}")

        if cannonized_config:
            goal_config = canonize_config(goal_config)

        return self.plan_and_moveJ(robot_name=robot_name,
                                   target_joints=goal_config,
                                   speed=speed,
                                   acceleration=acceleration,
                                   blend_radius=blend_radius,
                                   tolerance=tolerance,
                                   max_planning_time=max_planning_time,
                                   max_length_to_distance_ratio=max_length_to_distance_ratio)

    def moveL(self, robot_name, target_position, speed=0.1, tolerance=0.003, facing_down=True, max_steps=1000):
        self.zero_all_robots_vels_except(robot_name)

        current_joints = self.env.robots_joint_pos[robot_name]
        current_pose = self.motion_planner.get_forward_kinematics(robot_name, current_joints)
        goal_orientation = current_pose[0] if not facing_down else np.array(FACING_DOWN_R).flatten()
        start_pos = np.array(current_pose[1])
        end_pos = np.array(target_position)

        logging.info(f"MovingL {robot_name} from {start_pos} to {end_pos}")

        # Calculate direction and distance
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        unit_direction = direction / distance

        # Calculate step size based on speed
        step_size = speed * self.time_step

        steps_executed = 0
        while steps_executed < max_steps:
            current_pos = np.array(self.motion_planner.get_forward_kinematics(robot_name, current_joints)[1])
            remaining_vector = end_pos - current_pos
            remaining_distance = np.linalg.norm(remaining_vector)

            if remaining_distance < tolerance:
                return True

            # Calculate next position
            move_distance = min(step_size, remaining_distance)
            next_pos = current_pos + (remaining_vector / remaining_distance) * move_distance

            # Compute IK for the next position
            next_joints = self.motion_planner.ik_solve(robot_name, (goal_orientation, next_pos),
                                                       start_config=current_joints)
            if next_joints is None or len(next_joints) == 0:
                logging.warning(f" IK solution not found for {robot_name} at attempt {steps_executed + 1}")
                steps_executed += 1
                continue

            # Move the robot
            actions = {robot: self.env.robots_joint_pos[robot] for robot in self.env.robots_joint_pos if
                       robot != robot_name}
            actions[robot_name] = next_joints
            self.env.step(actions)



            # Update current joints
            current_joints = self.env.robots_joint_pos[robot_name]
            steps_executed += 1

        logging.warning(
            f"Linear movement for {robot_name} completed, but target not reached within tolerance after {max_steps} attempts")
        return False

    def zero_all_robots_vels_except(self, robot_name):
        for r in self.env.robots_joint_velocities.keys():
            if r != robot_name:
                self.env.set_robot_joints(r, self.env.robots_joint_pos[r], np.zeros(6), simulate_step=False)

    def reset(self, randomize=True, block_positions=None):
        state = self.env.reset(randomize=randomize, block_positions=block_positions)
        return state

    def activate_grasp(self, wait_steps=5, render_freq=8):
        self.env.set_gripper(True)
        self.motion_planner.attach_box_to_ee()
        return self.wait(wait_steps, render_freq=render_freq)

    def deactivate_grasp(self, wait_steps=5, render_freq=8):
        self.env.set_gripper(False)
        self.motion_planner.detach_box_from_ee()
        return self.wait(wait_steps, render_freq=render_freq)

    def wait(self, n_steps, render_freq=8):
        maintain_pos = self.env.robots_joint_pos
        for i in range(n_steps):
            self.env.step(maintain_pos)

    def facing_down_ik(self, agent, target_transform, max_tries=20):
        # Use inverse kinematics to get the joint configuration for this pose
        target_config = self.motion_planner.ik_solve(agent, target_transform)
        if target_config is None:
            target_config = []
        shoulder_constraint_for_down_movement = 0.1

        def valid_shoulder_angle(q):
            return -shoulder_constraint_for_down_movement > q[1] > -np.pi + shoulder_constraint_for_down_movement

        trial = 1
        while (self.motion_planner.is_config_feasible(agent, target_config) is False or
               valid_shoulder_angle(target_config) is False) \
                and trial < max_tries:
            trial += 1
            # try to find another solution, starting from other random configurations:
            q_near = np.random.uniform(-np.pi / 2, np.pi / 2, 6)
            target_config = self.motion_planner.ik_solve(agent, target_transform, start_config=q_near)
            if target_config is None:
                target_config = []

        return target_config

    def check_point_in_block(self, x, y):
        for block_id, pos in self.blocks_positions_dict.items():
            box_center = pos[:2].tolist()
            if point_in_square(square_center=box_center, edge_length=.04, point=[x, y]):
                return block_id
        return None

    def pick_up(self, agent, x, y, start_height=0.15):
        logging.info(f"pick up block at {x, y} start_height {start_height}")

        res = self.plan_and_move_to_xyz_facing_down(agent,
                                                    [x, y, start_height],
                                                    speed=4.,
                                                    acceleration=4.,
                                                    blend_radius=0.05,
                                                    tolerance=0.03, )
        if not res:
            return False

        above_block_config = self.env.robots_joint_pos[agent]

        self.moveL(agent,
                   (x, y, start_height-0.1),
                   speed=2.,
                   tolerance=0.003,
                   max_steps=400)
        self.wait(20)
        _ = self.activate_grasp()
        self.wait(5)
        self.moveJ(agent, above_block_config, speed=4., acceleration=4., tolerance=0.1)

        object_grasped = self.env.is_object_grasped()
        if not object_grasped:
            self.deactivate_grasp()

        return object_grasped

    def put_down(self, agent, x, y, start_height=0.15):
        release_height = self.env.get_tower_height_at_point((x, y)) + 0.04 + 0.025
        start_height = max(start_height, release_height + 0.05)

        logging.info(f"put down block at {x, y} start_height {start_height}")

        res = self.plan_and_move_to_xyz_facing_down(agent,
                                                    [x, y, start_height],
                                                    speed=4.,
                                                    acceleration=4.,
                                                    blend_radius=0.05,
                                                    tolerance=0.03, )
        if not res:
            return False

        above_block_config = self.env.robots_joint_pos[agent]

        self.moveL(agent,
                   (x, y, release_height),
                   speed=2.,
                   tolerance=0.003,
                   max_steps=400)
        self.wait(20)
        _ = self.deactivate_grasp()
        self.wait(20)
        self.moveJ(agent, above_block_config, speed=4., acceleration=4., tolerance=0.1)

    def sense_for_block(self, agent, x, y, start_height=0.15):
        logging.info(f"sense for block at {x, y} start_height {start_height}")

        res = self.plan_and_move_to_xyz_facing_down(agent,
                                                    [x, y, start_height],
                                                    speed=4.,
                                                    acceleration=4.,
                                                    blend_radius=0.05,
                                                    tolerance=0.03, )

        if not res:
            return False

        self.moveL(agent, (x, y, 0.05), speed=3., tolerance=0.003, max_steps=400)

        occupied = False

        for bpos in self.env.get_block_positions():
            if point_in_square(square_center=bpos[:2], edge_length=.04, point=[x, y]):
                occupied = True
                break

        self.moveL(agent, (x, y, start_height), speed=3., tolerance=0.005, max_steps=400)

        return occupied
