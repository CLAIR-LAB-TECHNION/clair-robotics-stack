import typer
from lab_setup.robot_inteface.robot_interface import RobotInterface
from lab_setup.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from motion_planning.geometry_and_transforms import GeometryAndTransforms


app = typer.Typer()


@app.command()
def main(robot_name="ur5e_2"):
    if robot_name == "ur5e_1":
        robot_data = ur5e_1
    elif robot_name == "ur5e_2":
        robot_data = ur5e_2
    else:
        raise ValueError("Invalid robot name")

    gt = GeometryAndTransforms.build()

    robot = RobotInterface(robot_data["ip"])
    # robot.freedriveMode()

    while True:
        pose = robot.getActualTCPPose()
        position_world = gt.point_robot_to_world(robot_data["name"], pose[:3])
        config = robot.getActualQ()

        print("--------------------------------------------")
        print("robot_name: ", robot_data["name"], "ip: ", robot_data["ip"])
        print("robot frame pose: ", pose)
        print("world frame position:", position_world)
        print("config: ", config)
        print("--------------------------------------------")


if __name__ == "__main__":
    app()
