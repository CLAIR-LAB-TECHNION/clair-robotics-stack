import random

import typer
from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_2
from ur_lab.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default


app = typer.Typer()

# add all corners of the workspace
points = [[workspace_x_lims_default[0], workspace_y_lims_default[0]],
            [workspace_x_lims_default[0], workspace_y_lims_default[1]],
            [workspace_x_lims_default[1], workspace_y_lims_default[0]],
            [workspace_x_lims_default[1], workspace_y_lims_default[1]]]


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(repeat: int = 1):
    controller = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])
    # controller.moveL_relative([0, 0, 0.1], speed=0.5, acceleration=0.5)
    controller.plan_and_move_home()
    for point in points:
        heights = []
        print("---tilted:")
        for i in range(repeat):
            h = controller.sense_height_tilted(point[0], point[1])
            print("measured height:", h)
            heights.append(h)

        mean = sum(heights) / len(heights)
        variance = sum((h - mean) ** 2 for h in heights) / len(heights)
        print("measured heights:", heights)
        print("mean:", mean, "variance:", variance)

    # try 20 more points in the workspace:
    for _ in range(100):
        point = [random.uniform(workspace_x_lims_default[0], workspace_x_lims_default[1]),
                 random.uniform(workspace_y_lims_default[0], workspace_y_lims_default[1])]
        for i in range(repeat):
            print("trying to sense height at:", point)
            h = controller.sense_height_tilted(point[0], point[1])
            print("measured height:", h)

    print("finished sensing heights")

if __name__ == "__main__":
    app()

