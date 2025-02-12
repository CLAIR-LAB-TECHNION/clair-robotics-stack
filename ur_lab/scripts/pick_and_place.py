import typer
from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_2


app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True}
)
def main(pickup_x: float,
         pickup_y: float,
         drop_x: float,
         drop_y: float,
         stacked_blocks: int = 1,
         back_and_forth_times: int = 1,
         speed_scale: float = 1.0,):
    typer.echo(f"Pickup location: {pickup_x, pickup_y}")
    typer.echo(f"Drop location: {drop_x, drop_y}")

    controller = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])
    controller.speed *= speed_scale
    controller.acceleration *= speed_scale

    # controller.move_home(speed=0.5, acceleration=0.5)

    for _ in range(back_and_forth_times):
        for _b in range(stacked_blocks):
            controller.pick_up(pickup_x, pickup_y, 0)
            controller.put_down(drop_x, drop_y, 0)

        # switch pickup and drop locations:
        pickup_x, pickup_y, drop_x, drop_y = drop_x, drop_y, pickup_x, pickup_y


if __name__ == "__main__":
    app()
