import numpy as np

from .transform import Transform

# UR5e dh parameters
# This can be find here:
# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
# or in the ur_e_description package
d = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
a = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
alpha = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]


# Forward Kinematics Function
def forward(joint_angles):
    T = np.eye(4)

    for i in range(6):
        A = np.array([
            [np.cos(joint_angles[i]), -np.sin(joint_angles[i]) * np.cos(alpha[i]),
             np.sin(joint_angles[i]) * np.sin(alpha[i]), a[i] * np.cos(joint_angles[i])],
            [np.sin(joint_angles[i]), np.cos(joint_angles[i]) * np.cos(alpha[i]),
             -np.cos(joint_angles[i]) * np.sin(alpha[i]), a[i] * np.sin(joint_angles[i])],
            [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])

        T = np.dot(T, A)

    return Transform.from_matrix(T)
