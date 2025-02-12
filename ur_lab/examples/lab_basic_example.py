from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_2


robot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])
robot.plan_and_move_home()

# rotate around base, which is the first joint, and move back:
q = robot.getActualQ()
q[0] -= 1.57
robot.moveJ(q, speed=0.5, acceleration=0.5)

# say hello:
hello_left = q.copy()
hello_right = q.copy()
hello_left[1] += 0.3
hello_left[2] += 0.4
hello_right[1] -= 0.3
hello_right[2] -= 0.4
path = [hello_left, hello_right, hello_left, hello_right, q]
print("hello!")
robot.move_path(path, speed=2., acceleration=1., blend_radius=0.3)


# go back home
robot.plan_and_move_home()
