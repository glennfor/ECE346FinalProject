"""
Publishes ILQR plan trajectories as nav_msgs/Path for RViz visualization.

Usage:
    1. Create a Visualizer in the ROS node, passing the node itself.
    2. Call update_planner_plan(trajectory) / update_monitor_plan(trajectory)
       whenever a new plan is available.
    3. The visualizer publishes on /Planning/Trajectory (green in default
       RViz config) and /SafetyFilter/MonitorPlan.
"""

import math

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node


class Visualizer:

    def __init__(self, node: Node, frame_id: str = 'map',
                 planner_topic: str = '/Planning/Trajectory',
                 monitor_topic: str = '/SafetyFilter/MonitorPlan'):
        self._node = node
        self._frame_id = frame_id
        self._planner_pub = node.create_publisher(PathMsg, planner_topic, 1)
        self._monitor_pub = node.create_publisher(PathMsg, monitor_topic, 1)

    def update_planner_plan(self, trajectory: np.ndarray):
        """Publish the planner ILQR trajectory (5, T) as a Path."""
        if trajectory is None or trajectory.size == 0:
            return
        self._planner_pub.publish(self._to_path_msg(trajectory))

    def update_monitor_plan(self, trajectory: np.ndarray):
        """Publish the monitor ILQR trajectory (5, T) as a Path."""
        if trajectory is None or trajectory.size == 0:
            return
        self._monitor_pub.publish(self._to_path_msg(trajectory))

    def _to_path_msg(self, trajectory: np.ndarray) -> PathMsg:
        path = PathMsg()
        path.header.stamp = self._node.get_clock().now().to_msg()
        path.header.frame_id = self._frame_id

        T = trajectory.shape[1]
        for t in range(T):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(trajectory[0, t])
            pose.pose.position.y = float(trajectory[1, t])
            pose.pose.position.z = 0.0
            psi = float(trajectory[3, t])
            pose.pose.orientation.z = math.sin(psi / 2.0)
            pose.pose.orientation.w = math.cos(psi / 2.0)
            path.poses.append(pose)
        return path