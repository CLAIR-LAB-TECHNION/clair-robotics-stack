import os
import time
import logging
import pandas as pd
import typer
import numpy as np
import chime

from experiments_lab.block_stacking_env import LabBlockStackingEnv
from ur_lab.motion_planning.motion_planner import MotionPlanner
from ur_lab.motion_planning.geometry_and_transforms import GeometryAndTransforms
from ur_lab.manipulation.manipulation_controller_2fg import ManipulationController2FG
from ur_lab.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from ur_lab.camera.realsense_camera import RealsenseCameraWithRecording
from ur_lab.vision.image_block_position_estimator import ImageBlockPositionEstimator
from ur_lab.utils.workspace_utils import (workspace_x_lims_default,
                                          workspace_y_lims_default, goal_tower_position)
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.pouct_planner_policy import POUCTPolicy
from modeling.policies.hand_made_policy import HandMadePolicy
from experiments_lab.experiment_manager import ExperimentManager



camera = RealsenseCameraWithRecording()

motion_planner = MotionPlanner()
gt = GeometryAndTransforms.from_motion_planner(motion_planner)
position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)

r1_controller.speed, r1_controller.acceleration = 0.75, 0.75


record_dir = "voa-pomdp/lab_ur_stack/camera/recordings"
print("ready to record")
camera.start_recording(record_dir)
time.sleep(60)
camera.stop_recording()




