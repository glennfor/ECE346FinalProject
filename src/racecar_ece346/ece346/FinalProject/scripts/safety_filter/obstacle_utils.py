import numpy as np
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker


def get_obstacle_vertices(obs: Marker):
    """
    Convert a cubic Marker to its 8 vertices in the world frame.

    Returns:
        (id, vertices_global): obstacle id and (8, 3) numpy array
    """
    x_size = obs.scale.x
    y_size = obs.scale.y
    z_size = obs.scale.z

    vertices_local = np.array([
        [-x_size / 2, -y_size / 2, -z_size / 2, 1],
        [-x_size / 2, -y_size / 2,  z_size / 2, 1],
        [-x_size / 2,  y_size / 2, -z_size / 2, 1],
        [-x_size / 2,  y_size / 2,  z_size / 2, 1],
        [ x_size / 2, -y_size / 2, -z_size / 2, 1],
        [ x_size / 2, -y_size / 2,  z_size / 2, 1],
        [ x_size / 2,  y_size / 2, -z_size / 2, 1],
        [ x_size / 2,  y_size / 2,  z_size / 2, 1],
    ]).T

    orientation = obs.pose.orientation
    position = obs.pose.position
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    trans = [position.x, position.y, position.z]

    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = trans

    vertices_global = np.dot(T, vertices_local)
    return obs.id, vertices_global[:3, :].T
