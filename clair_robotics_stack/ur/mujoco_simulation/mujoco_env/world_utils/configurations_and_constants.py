import numpy as np

from clair_robotics_stack.ur.mujoco_simulation.mujoco_env.tasks.null_task import NullTask

import os

import numpy as np
from clair_robotics_stack.ur.mujoco_simulation.mujoco_env.episode.specs.scene_spec import ObjectSpec
from clair_robotics_stack.ur.mujoco_simulation.mujoco_env.episode.specs.joint_spec import JointSpec
# from gymjoco.tasks import NullTask
# from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld
# from n_table_blocks_world.configurations_and_constants import ROBOTIQ_2F85_BODY
# from tamp_helpers.table_sampling import sample_on_table


#Use in sim_env.py, at init


INIT_MAX_VELOCITY = np.array([3]*6)

# relative position of grasped object from end effector
grasp_offset = 0.02
frame_skip = 5



muj_env_config = dict(
    scene=dict(
        resource='clairlab',
        render_camera='top-right',
    ),
    robots=dict(
        ur5e_1=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[0, 0, 0.01],
            base_rot=[0, 0, 1.57079632679],
            privileged_info=True,
        ),
        ur5e_2=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[-0.76, -1.33, 0.01],
            base_rot=[0, 0, -1.57079632679],
            privileged_info=True,
        ),
    ),
    tasks=dict(
        ur5e_1=NullTask,
        ur5e_2=NullTask,
    ),
)



muj_env_with_costume_config = dict(
    scene=dict(
        resource='costume',
        render_camera='top-right',
        objects=ObjectSpec('can', base_joints=JointSpec('free'), base_pos=[-0.7, -0.6, 0.03]),
        objects=ObjectSpec('can', base_joints=JointSpec('free'), base_pos=[-0.73, -0.6, 0.03]),
    ),
    robots=dict(
        ur5e_1=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[0, 0, 0.01],
            base_rot=[0, 0, 1.57079632679],
            privileged_info=True,
        ),
        ur5e_2=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[-0.76, -1.33, 0.01],
            base_rot=[0, 0, -1.57079632679],
            privileged_info=True,
        ),
    ),
    tasks=dict(
        ur5e_1=NullTask,
        ur5e_2=NullTask,
    ),
)



muj_env_for_s3e_config = dict(
    scene=dict(
        resource='s3e',
        resource='s3e',
        render_camera='top-right',
        objects=(
            ObjectSpec('bread', base_joints=JointSpec('free'), base_pos=[-0.7, -0.6, 0.03]),
            ObjectSpec('can', base_joints=JointSpec('free'), base_pos=[-0.7, -0.7, 0.03]),
            # ObjectSpec('can', base_joints=JointSpec('free'), base_pos=[-0.205, -0.3, 0.03]),  
                 ),
    ),
    robots=dict(
        ur5e_1=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[0, 0, 0.01],
            base_rot=[0, 0, 1.57079632679],
            privileged_info=True,
        ),
        ur5e_2=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[-0.76, -1.33, 0.01],
            base_rot=[0, 0, -1.57079632679],
            privileged_info=True,
        ),
    ),
    tasks=dict(
        ur5e_1=NullTask,
        ur5e_2=NullTask,
    ),
)
