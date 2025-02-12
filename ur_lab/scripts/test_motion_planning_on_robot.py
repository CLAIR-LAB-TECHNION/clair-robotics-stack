from ur_lab.robot_inteface.robot_interface import RobotInterfaceWithGripper
import time
from ur_lab.motion_planning.motion_planner import MotionPlanner
from ur_lab.motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.robot_inteface.robots_metadata import ur5e_1, ur5e_2

target_position_world_rob1 = [-0.3, -0.5, 0.25]
target_position_world_rob2 = [-0.4, -0.7, 0.25]


# robot1 = RobotInterfaceWithGripper(ur5e_1["ip"], 50)
robot2 = RobotInterfaceWithGripper(ur5e_2["ip"], 50)

motion_planner = MotionPlanner()
motion_planner.visualize()

gt = GeometryAndTransforms.from_motion_planner(motion_planner)
target_pose_rob1_local = gt.get_gripper_facing_downwards_6d_pose_robot_frame(ur5e_1["name"],
                                                                             target_position_world_rob1,
                                                                             0)
target_pose_rob2_local = gt.get_gripper_facing_downwards_6d_pose_robot_frame(ur5e_2["name"],
                                                                             target_position_world_rob2,
                                                                             0)

time.sleep(0.2)
# robot1.move_home(speed=0.5, acceleration=0.5, asynchronous=True)
robot2.move_home(speed=0.5, acceleration=0.5, asynchronous=False)
time.sleep(0.5)

init_config = robot2.getActualQ()

motion_planner.show_point_vis(target_position_world_rob2)

target_config = robot2.getInverseKinematics(target_pose_rob2_local)
print("target_config: ", target_config)

motion_planner.ur5e_2.setConfig(motion_planner.config6d_to_klampt(target_config))

path = motion_planner.plan_from_start_to_goal_config("ur5e_2", init_config, target_config)
motion_planner.vis_path("ur5e_2", path)

vel, acc, blend = 0.5, 1., 0.01
path = [[*target_config, vel, acc, blend] for target_config in path]
robot2.moveJ(path)

######################
#
# init_config = robot1.getActualQ()
#
# motion_planner.show_point_vis(target_position_world_rob1)
# target_config = robot1.getInverseKinematics(target_pose_rob1_local)
# print("target_config: ", target_config)
# motion_planner.ur5e_1.setConfig(motion_planner.config6d_to_klampt(target_config))
#
# path = motion_planner.plan_from_start_to_goal_config("ur5e_1", init_config, target_config)
# motion_planner.show_path_vis("ur5e_1", path)
#
# vel, acc, blend = 0.5, 1., 0.01
# path = [[*target_config, vel, acc, blend] for target_config in path]
# robot1.moveJ(path)
