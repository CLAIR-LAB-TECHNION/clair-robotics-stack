import numpy as np

from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms


def test_world_transforms():
    max_error = 1e-3
    motion_planner = MotionPlanner()
    transforms = GeometryAndTransforms.from_motion_planner(motion_planner)

    # test first robot's transforms. The robot should be aligned with the world coordinate system.
    world_point = [0, 0, 0]
    robot_point = transforms.point_world_to_robot("ur5e_1", world_point)
    assert robot_point == world_point, f"robot_point: {robot_point}, world_point: {world_point}"

    world_point= [0.1, 0.2, 0.3]
    robot_point = transforms.point_world_to_robot("ur5e_1", world_point)
    assert robot_point == world_point, f"robot_point: {robot_point}, world_point: {world_point}"

    robot_point = [0, 0, 0]
    world_point = transforms.point_robot_to_world("ur5e_1", robot_point)
    assert world_point == robot_point, f"world_point: {world_point}, robot_point: {robot_point}"

    robot_point = [0.1, 0.2, 0.3]
    world_point = transforms.point_robot_to_world("ur5e_1", robot_point)
    assert world_point == robot_point, f"world_point: {world_point}, robot_point: {robot_point}"

    # test second robot's transforms
    world_point = [0, 0, 0]
    robot_point = transforms.point_world_to_robot("ur5e_2", world_point)
    expected_robot_point = [-0.765, -1.33, 0]
    error = np.linalg.norm(np.array(robot_point) - np.array(expected_robot_point))
    assert error < max_error, f"robot_point: {robot_point}, expected_robot_point: {expected_robot_point}"

    world_point = (-0.5, -0.5, -0.5)
    robot_point = transforms.point_world_to_robot("ur5e_2", world_point)
    expected_robot_point = [-0.265, -0.83, -0.5]
    error = np.linalg.norm(np.array(robot_point) - np.array(expected_robot_point))
    assert error < max_error, f"robot_point: {robot_point}, expected_robot_point: {expected_robot_point}"

    robot_point = [0, 0, 0]
    world_point = transforms.point_robot_to_world("ur5e_2", robot_point)
    expected_world_point = [-0.765, -1.33, 0]
    error = np.linalg.norm(np.array(world_point) - np.array(expected_world_point))
    assert error < max_error, f"world_point: {world_point}, expected_world_point: {expected_world_point}"

    robot_point = [-0.5, -0.5, -0.5]
    world_point = transforms.point_robot_to_world("ur5e_2", robot_point)
    expected_world_point = [-0.265, -0.83, -0.5]
    error = np.linalg.norm(np.array(world_point) - np.array(expected_world_point))
    assert error < max_error, f"world_point: {world_point}, expected_world_point: {expected_world_point}"


if __name__ == "__main__":
    test_world_transforms()
    print("test_world_transforms passed")
