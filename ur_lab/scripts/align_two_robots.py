import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_1, ur5e_2


app = typer.Typer()

mid_point = [-0.4, -0.6]
offset = 0.08


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(axis="x",):

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController2FG(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed = 0.3
    r1_controller.acceleration = 0.3
    r2_controller.speed = 0.3
    r2_controller.acceleration = 0.3

    r1_point = mid_point.copy()
    r2_point = mid_point.copy()

    if axis == "x":
        r1_point[0] += offset
        r2_point[0] -= offset
    elif axis == "y":
        r1_point[1] += offset
        r2_point[1] -= offset

    r1_controller.plan_and_move_to_xyzrz(r1_point[0], r1_point[1], 0.2, 0)
    r2_controller.plan_and_move_to_xyzrz(r2_point[0], r2_point[1], 0.2, 0)


if __name__ == "__main__":
    app()

