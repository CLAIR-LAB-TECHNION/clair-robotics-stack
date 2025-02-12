from ur_lab.robot_inteface.robot_interface import RobotInterfaceWithSuctionGripper
from ur_lab.robot_inteface.robots_metadata import ur5e_1
import time
from ur_lab.manipulation.manipulation_controller_vg import ManipulationControllerVG
import numpy as np

controller = ManipulationControllerVG.build_from_robot_name_and_ip(ur5e_1["ip"], ur5e_1["name"])
controller.speed = 0.5
controller.pick_up(0.3, -0.3, -np.pi/4, start_height=0.2, channel=2, vaccum_power=60)
controller.put_down(0.3, -0.2, -np.pi/4, start_height=0.2, channel=2)




# start = [1.4833663702011108,
#  -2.0520292721190394,
#  1.9133380095111292,
#  -1.4182146352580567,
#  -1.5658367315875452,
#  2.2819480895996094-3.14]

# robot = RobotInterfaceWithSuctionGripper(ur5e_1["ip"])
# robot.moveJ(start)
#
# # pick
# robot.moveUntilContact(xd=[0, 0, -0.05, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
# robot.moveL_relative([0, 0, -0.01], speed=0.1) #
# time.sleep(0.2)
# robot.suction.vg10c_grip(channel=2, vacuum_power=60.)
# time.sleep(0.2)
# robot.moveL_relative([0, 0, 0.05], speed=0.1)
#
# # put
# robot.moveUntilContact(xd=[0, 0, -0.02, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
# robot.moveL_relative([0, 0, -0.001], speed=0.1)
# robot.suction.vg10c_release(channel=2)
# time.sleep(0.2)
# robot.moveL_relative([0, 0, 0.05], speed=0.1)

