import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from scipy.spatial.transform import Rotation as R


@dataclass
class PoseE:
    x: float
    y: float
    z: float
    Rx: float
    Ry: float
    Rz: float

    def __iter__(self):
        # Yield attributes in the order you want them unpacked
        yield self.x
        yield self.y
        yield self.z
        yield self.Rx
        yield self.Ry
        yield self.Rz

    def tolist(self) -> List[float]:
        return [self.x, self.y, self.z, self.Rx, self.Ry, self.Rz]

    def translation(self) -> Tuple[float, float, float]:
        """
        Returns the translation components of the pose.

        :return: Tuple containing (x, y, z).
        """
        return self.x, self.y, self.z

    def angles(self) -> Tuple[float, float, float]:
        """
        Returns the Euler angles of the pose.

        :return: Tuple containing (Rx, Ry, Rz).
        """
        return self.Rx, self.Ry, self.Rz


@dataclass
class PoseQ:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float  # Adding the scalar part of the quaternion

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
        yield self.qx
        yield self.qy
        yield self.qz
        yield self.qw

    def tolist(self) -> List[float]:
        return [self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]

    def translation(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def quaternion(self) -> Tuple[float, float, float, float]:
        return self.qx, self.qy, self.qz, self.qw


class Point3D(np.ndarray):
    def __new__(cls, x, y, z):
        obj = np.asarray([x, y, z], dtype=float).view(cls)
        return obj

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"


class Transform:
    def __init__(self, rotation=np.eye(3), translation=np.zeros(3), from_frame='world', to_frame='object'):
        self.rotation = rotation
        self.translation = translation
        self.from_frame = from_frame
        self.to_frame = to_frame
        if not self.is_valid():
            raise ValueError("Invalid transformation parameters.")

    def is_valid(self) -> bool:
        if not np.allclose(np.dot(self.rotation, self.rotation.T), np.eye(3), atol=1e-6):
            return False

        if not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6):
            return False

        if self.translation.shape != (3,) or not np.issubdtype(self.translation.dtype, np.number):
            return False

        return True

    def __str__(self):
        return (f"Transform from '{self.from_frame}' to '{self.to_frame}':\n"
                f"Rotation:\n{self.rotation}\n"
                f"Translation: {self.translation}")

    def __repr__(self):
        return (f"Transform(rotation={self.rotation}, "
                f"translation={self.translation}, "
                f"from_frame='{self.from_frame}', "
                f"to_frame='{self.to_frame}')")

    def inverse(self) -> 'Transform':
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return Transform(inv_rotation, inv_translation, self.to_frame, self.from_frame)

    @staticmethod
    def rotation_matrix_x(angle: float):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return Transform(rotation=rotation)

    @staticmethod
    def rotation_matrix_y(angle: float):
        c, s = np.cos(angle), np.sin(angle)
        return Transform(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))

    @staticmethod
    def rotation_matrix_z(angle: float):
        c, s = np.cos(angle), np.sin(angle)
        return Transform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

    @classmethod
    def from_pose_quaternion(cls, pose: PoseQ) -> 'Transform':
        """
        Create a Transform object from a pose with position and quaternion.
        """
        translation = np.array([pose.x, pose.y, pose.z])
        rotation = R.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        return cls(rotation=rotation, translation=translation)

    def to_pose_quaternion(self) -> PoseQ:
        """
        Convert the Transform object back to a Pose object using quaternion.
        """
        quaternion = R.from_matrix(self.rotation).as_quat(canonical=True)
        return PoseQ(self.translation[0], self.translation[1], self.translation[2], quaternion[0], quaternion[1],
                     quaternion[2], quaternion[3])

    def adjust_for_camera_pose(self) -> 'Transform':

        adjusted_rotation = self.rotation.copy()
        adjusted_rotation[:, 2] = -adjusted_rotation[:, 2]
        adjusted_rotation[:, 1] = -adjusted_rotation[:, 1]

        return Transform(rotation=adjusted_rotation, translation=self.translation, from_frame=self.from_frame,
                         to_frame=self.to_frame)

    def adjust_to_look_at_format(self) -> 'Transform':

        adjusted_rotation = self.rotation.copy()
        adjusted_rotation[:, 2] = -adjusted_rotation[:, 2]
        adjusted_rotation[:, 1] = -adjusted_rotation[:, 1]

        return Transform(rotation=adjusted_rotation, translation=self.translation, from_frame=self.from_frame,
                         to_frame=self.to_frame)

    def get_transformation_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    def compose(self, other_transform: 'Transform') -> 'Transform':
        new_rotation = self.rotation @ other_transform.rotation
        new_translation = self.rotation @ other_transform.translation + self.translation
        return Transform(new_rotation, new_translation, self.from_frame, other_transform.to_frame)

    def apply_to_mesh(self, mesh):
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(self.get_transformation_matrix())
        return transformed_mesh

    def to_pose_zyx(self) -> PoseE:
        x, y, z = self.translation

        R = self.rotation
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6  # Check for singularity

        if not singular:
            Rx = np.arctan2(R[2, 1], R[2, 2])
            Ry = np.arctan2(-R[2, 0], sy)
            Rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            Rx = np.arctan2(-R[1, 2], R[1, 1])
            Ry = np.arctan2(-R[2, 0], sy)
            Rz = 0

        return PoseE(x, y, z, Rx, Ry, Rz)

    def to_pose_xyz(self) -> PoseE:
        x, y, z = self.translation

        R = self.rotation

        singular = np.abs(R[0, 2]) > 0.99999

        if not singular:
            Ry = np.arcsin(R[0, 2])
            Rx = np.arctan2(-R[1, 2] / np.cos(Ry), R[2, 2] / np.cos(Ry))
            Rz = np.arctan2(-R[0, 1] / np.cos(Ry), R[0, 0] / np.cos(Ry))
        else:
            Rz = 0
            Ry = np.pi / 2 * np.sign(R[0, 2])
            Rx = np.arctan2(R[1, 0], R[1, 1])

        return PoseE(x, y, z, Rx, Ry, Rz)

    def to_pose_axis_angle(self) -> PoseE:
        x, y, z = self.translation  # Translation components
        R = self.rotation  # Rotation matrix

        theta = np.arccos((np.trace(R) - 1) / 2)  # Rotation angle
        # Rodrigues' formula for rotation axis
        r = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))

        # Rotation vector
        rv = r * theta
        Rx = rv[0]
        Ry = rv[1]
        Rz = rv[2]

        return PoseE(x, y, z, Rx, Ry, Rz)

    @staticmethod
    def _euler_to_rotation_matrix(Rx: float, Ry: float, Rz: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Euler angles to a rotation matrix.

        :param Rx, Ry, Rz: Euler angles in radians.
        :return: A 3x3 rotation matrix.
        """
        Rx_matrix = np.array([[1, 0, 0],
                              [0, np.cos(Rx), -np.sin(Rx)],
                              [0, np.sin(Rx), np.cos(Rx)]])

        Ry_matrix = np.array([[np.cos(Ry), 0, np.sin(Ry)],
                              [0, 1, 0],
                              [-np.sin(Ry), 0, np.cos(Ry)]])

        Rz_matrix = np.array([[np.cos(Rz), -np.sin(Rz), 0],
                              [np.sin(Rz), np.cos(Rz), 0],
                              [0, 0, 1]])

        # XYZ rotation matrix
        xyz_matrix = Rx_matrix @ Ry_matrix @ Rz_matrix

        # ZXY rotation matrix
        zxy_matrix = Rz_matrix @ Rx_matrix @ Ry_matrix

        return xyz_matrix, zxy_matrix

    @classmethod
    def from_pose_zyx(cls, pose: PoseE) -> 'Transform':
        """
        Create a Transform object from a pose represented by (x, y, z, Rx, Ry, Rz).

        :param pose: Pose containing position and Euler angles (x, y, z, Rx, Ry, Rz).
        :return: A Transform object with the corresponding rotation and translation.
        """
        x, y, z, Rx, Ry, Rz = pose

        _, rotation_matrix = cls._euler_to_rotation_matrix(Rx, Ry, Rz)

        translation_vector = np.array([x, y, z])

        return cls(rotation=rotation_matrix, translation=translation_vector)

    @classmethod
    def from_pose_xyz(cls, pose: PoseE) -> 'Transform':
        """
        Create a Transform object from a pose represented by (x, y, z, Rx, Ry, Rz).

        :param pose: Pose containing position and Euler angles (x, y, z, Rx, Ry, Rz).
        :return: A Transform object with the corresponding rotation and translation.
        """
        x, y, z, Rx, Ry, Rz = pose

        rotation_matrix, _ = cls._euler_to_rotation_matrix(Rx, Ry, Rz)

        translation_vector = np.array([x, y, z])

        return cls(rotation=rotation_matrix, translation=translation_vector)

    @classmethod
    def from_rv(cls, pose: PoseE, from_frame: str = 'world', to_frame: str = 'EE') -> 'Transform':

        rotation_matrix = R.from_rotvec(pose.angles()).as_matrix()
        translation = np.array(pose.translation())
        return cls(rotation=rotation_matrix, translation=translation, from_frame=from_frame, to_frame=to_frame)

    @classmethod
    def from_matrix(cls, matrix, from_frame='world', to_frame='object') -> 'Transform':
        """Create a Transform object from a 4x4 transformation matrix."""
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be a 4x4 transformation matrix.")

        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        return cls(rotation=rotation, translation=translation, from_frame=from_frame, to_frame=to_frame)
