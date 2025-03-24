import typer
from ur_lab.robot_inteface.robot_interface import RobotInterface
from ur_lab.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from motion_planning.geometry_and_transforms import GeometryAndTransforms


app = typer.Typer()


@app.command()
def main(robot_name="ur5e_1"):
    if robot_name == "ur5e_1":
        robot_data = ur5e_1
    elif robot_name == "ur5e_2":
        robot_data = ur5e_2
    else:
        raise ValueError("Invalid robot name")

    gt = GeometryAndTransforms.build()

    robot = RobotInterface(robot_data["ip"])
    robot.freedriveMode()

    # save robot config everytime enter is pressed and end with 'E'
    configs = []
    while True:
        input("Press enter to save current config")
        config = robot.getActualQ()
        configs.append(config)
        print("config: ", config)
        print("Press 'E' to exit")
        if input() == 'E':
            break

    robot.endFreedriveMode()

    print("--------------------")
    print("all configs: ",)
    for config in configs:
        print(config)


if __name__ == "__main__":
    app()
