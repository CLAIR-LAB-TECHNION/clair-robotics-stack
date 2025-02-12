import typer

from ur_lab.motion_planning.motion_planner import MotionPlanner
from ur_lab.motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from ur_lab.utils.workspace_utils import (workspace_x_lims_default,
                                          workspace_y_lims_default, stack_position_r2frame)

mid_workspace = [(workspace_x_lims_default[0] + workspace_x_lims_default[1]) / 2,
                 (workspace_y_lims_default[0] + workspace_y_lims_default[1]) / 2]

app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(offset_x: float = -0.015,
         offset_y: float = -0.02):
    r2_controller = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])
    gt = GeometryAndTransforms.from_motion_planner(MotionPlanner())

    stack_position = gt.point_robot_to_world("ur5e_2", (*stack_position_r2frame, 0))
    stack_position = stack_position[:2]

    r2_controller.pick_up(stack_position[0], stack_position[1], 0, 0.25)
    r2_controller.put_down(mid_workspace[0], mid_workspace[1], 0, 0.1)

    # try to pick up with offset:
    r2_controller.pick_up(mid_workspace[0] + offset_x, mid_workspace[1] + offset_y, 0, 0.1)

    # put back down:
    r2_controller.put_down(mid_workspace[0], mid_workspace[1], 0, 0.1)


if __name__ == "__main__":
    app()
