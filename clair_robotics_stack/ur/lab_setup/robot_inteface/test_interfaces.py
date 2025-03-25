from lab_setup.robot_inteface.robot_interface import RobotInterfaceWithGripper, home_config
import numpy
from numpy import pi
import time


robot1 = RobotInterfaceWithGripper("192.168.0.10", 125)
robot2 = RobotInterfaceWithGripper("192.168.0.11", 125)
time.sleep(0.5)

robots = [robot1, robot2]

for robot in robots:
    robot.move_home()
    robot.release_grasp()

for robot in robots:
        target1 = home_config.copy()
        target2 = home_config.copy()
        target3 = home_config.copy()
        target1[0] += pi/6
        target3[2] += pi/4
        vel, acc, blend = 1., 1., 0.001
        path = [[*target1, vel, acc, blend],
                [*target2, vel, acc, blend],
                [*target3, vel, acc, blend]]
        robot.moveJ(path, asynchronous=True)
time.sleep(0.3)


while True:
        robots_vel_norm = [numpy.linalg.norm(robot.getActualQd()) for robot in robots]
        if all([vel < 0.1 for vel in robots_vel_norm]):
            break
        time.sleep(0.1)

for robot in robots:
        robot.stopJ()
        pose = robot.getActualTCPPose()
        pose[2] -= 0.1
        robot.moveJ_IK(pose, speed=0.5, acceleration=0.5)
        robot.grasp()

        robot.move_home()


