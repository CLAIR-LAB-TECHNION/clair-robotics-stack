import matplotlib.pyplot as plt

import cv2
from ur_lab.motion_planning.motion_planner import MotionPlanner
from ur_lab.motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_1
from ur_lab.camera.realsense_camera import RealsenseCamera

if __name__ == "__main__":
    camera = RealsenseCamera()

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    # r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)

    # r2_controller.move_home()
    # r1_controller.plan_and_moveJ([-0.5, -np.pi/2, 0, -np.pi/2, 0, 0])

    # r2_ee_position = motion_planner.get_forward_kinematics("ur5e_2", r2_controller.getActualQ())[1]
    # print("r2_ee_position according to fk:", r2_ee_position)

    # r2_ee_position = r2_controller.getActualTCPPose()[:3]
    # r2_ee_position = gt.point_robot_to_world("ur5e_2", r2_ee_position)
    # point_world = r2_ee_position

    point_world = [0.985, -0.415, -0.01]

    print("point_world:", point_world)

    configs = [
        # [-0.03570539156068975, -2.080632825891012, -0.04940330982208252, -2.100678106347555, 1.474133014678955, 0.2835099399089813],
        #        [0.30966347455978394, -2.7083732090392054, -0.08799216896295547, -2.1580645046629847, 0.8967615962028503,
        #         0.28159618377685547],
        #        [-0.8021147886859339, -2.5786100826659144, -0.04320277273654938, -1.814143796960348, 2.445481061935425,
        #         0.28170034289360046],
        #        [-0.21004897752870733, -2.55362667659902, -0.043249256908893585, -2.147140165368551, 1.5722019672393799,
        #         0.02884768135845661],
               [-0.2923691908465784, -2.8978945217528285, -0.04456430673599243, -1.5825473270811976, 1.580686092376709,
                0.028799772262573242],
               [-0.24980527559389287, -2.8275796375670375, -0.03361622244119644, -1.377223090534546, 1.3678569793701172,
                -1.8001263777362269],
               [-0.2706406752215784, -2.9091884098448695, -0.11053931713104248, -1.573202749291891, 1.4076406955718994,
                -0.10053855577577764]
    ]
    for c in configs:
        r1_controller.plan_and_moveJ(c, speed=0.7, acceleration=0.7)
        point_camera = gt.point_world_to_camera(point_world, "ur5e_1", r1_controller.getActualQ(), r1_controller)
        print("point_camera:", point_camera)

        from ur_lab.camera.configurations_and_params import color_camera_intrinsic_matrix

        point_image_homogenous = color_camera_intrinsic_matrix @ point_camera
        point_image = point_image_homogenous / point_image_homogenous[2]
        print("point_image:", point_image)

        image, _ = camera.get_frame_bgr()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.scatter(point_image[0], point_image[1], c='r')
        plt.show()

