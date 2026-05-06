#!/usr/bin/env python3
"""
Final Project — Safety Filter (STUDENT SKELETON)

You will build a ROS2 node that sits between a human driver (PS4 joystick)
and the vehicle, intervening only when necessary to keep the car safe.

Pipeline:

    /joy --> joy_to_ackermann --> /teleop ┐
                                           │
                             /SLAM/Pose ───┤---> [safety_filter_node] ---> /drive
                                           │
                       /Obstacles/Static ──┘

Topics you will work with:

    /teleop            ackermann_msgs/AckermannDriveStamped  (human command in)
    <odom_topic>       nav_msgs/Odometry                     (vehicle state in)
    /Obstacles/Static  visualization_msgs/MarkerArray        (obstacles in)
    /drive             ackermann_msgs/AckermannDriveStamped  (safe command out)

------------------------------------------------------------------------------
Task 1: Build the node plumbing (subscribers, publisher, timer).
Task 2: Implement the safety filter logic.
------------------------------------------------------------------------------

You may:
  - Add imports, helper methods, and ROS parameters to THIS file.
  - Add your own files anywhere under FinalProject/ — e.g. an `ilqr/`
    subfolder with your ILQR solver, an `ilqr_params.yaml` with cost
    weights, utility modules, etc. Load yaml configs from within this
    node using `open()` + `yaml.safe_load()`, or declare a ROS param
    for the config path and set it in `final_project_*.yaml`.
  - Edit any yaml under `FinalProject/config/` — add parameters, tune
    thresholds, change topic names. All existing yaml values are
    documented and intended to be tunable.

You should NOT need to rewrite the launch files, the plumbing nodes
(joy_to_ackermann, drive_to_servo, etc.), or the CMakeLists top-level
install lists. Only touch those if you are adding a new standalone
executable — in which case ask first.
"""

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from visualization_msgs.msg import MarkerArray

from ament_index_python.packages import get_package_share_directory

from ece346.FinalProject.ILQR_Example.ref_path import RefPath
from ece346.FinalProject.ILQR_Example.ilqr import ILQR
from ece346.FinalProject.scripts.safety_filter.obstacle_utils import get_obstacle_vertices
from ece346.FinalProject.scripts.safety_filter.projector import ForwardProjector
from ece346.FinalProject.scripts.safety_filter.cost_evaluator import CostEvaluator


def yaw_from_quat(qx, qy, qz, qw):
    """Extract yaw (heading, rad) from a quaternion. Useful for Task 2."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SafetyFilterNode(Node):

    # =========================================================================
    # TASK 1 — Node setup (subscribers, publisher, timer)
    # =========================================================================
    #
    # Fill in __init__ below so that the node:
    #   1. Declares ROS parameters for each topic name and for the publish rate.
    #      Hint: use self.declare_parameter('name', default_value). The yaml
    #      file (final_project_*.yaml) will override these at launch time.
    #      Required parameter names (match the yaml):
    #          teleop_topic, drive_topic, odom_topic, static_obs_topic,
    #          publish_rate
    #
    #   2. Creates three subscribers that cache the latest message each:
    #          /teleop            -> AckermannDriveStamped
    #          <odom_topic>       -> Odometry
    #          /Obstacles/Static  -> MarkerArray
    #      Hint: self.create_subscription(MsgType, topic, callback, queue_size)
    #      Each callback can be a one-liner that stores msg into an instance
    #      attribute (e.g. self._latest_teleop = msg).
    #
    #   3. Creates one publisher:
    #          /drive             -> AckermannDriveStamped
    #      Hint: self.create_publisher(MsgType, topic, queue_size)
    #
    #   4. Creates a timer at `publish_rate` Hz that calls a method which
    #      invokes self.safety_filter(...) and publishes the result.
    #      Hint: self.create_timer(period_sec, callback)
    #
    # For reference, open any other node in this repo (e.g.
    # FinalProject/scripts/joy_to_ackermann_node.py) to see the same pattern.
    # =========================================================================

    def __init__(self):
        super().__init__('safety_filter_node')

        # ---- TODO(Task 1.1): declare ROS parameters ----
        self.declare_parameter('teleop_topic', '/teleop')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('odom_topic', '/slam_pose')
        self.declare_parameter('static_obs_topic', '/Obstacles/Static')
        self.declare_parameter('routing_path_topic', '/Routing/Path')
        self.declare_parameter('publish_rate', 30.0)

        self.declare_parameter('projection_horizon_sec', 2.0)
        self.declare_parameter('projection_dt', 0.2)
        self.declare_parameter('projection_min_speed', 0.5)

        self.declare_parameter('obs_soft_threshold', 5.0)
        self.declare_parameter('obs_hard_threshold', 50.0)
        self.declare_parameter('lane_soft_threshold', 300.0)
        self.declare_parameter('lane_hard_threshold', 1000.0)
        self.declare_parameter('total_soft_threshold', 400.0)
        self.declare_parameter('total_hard_threshold', 2000.0)

        self.declare_parameter('wheelbase', 0.324)

        # ---- TODO(Task 1.2): read parameter values ----
        teleop_topic = self.get_parameter('teleop_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        obs_topic = self.get_parameter('static_obs_topic').value
        path_topic = self.get_parameter('routing_path_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        self._proj_horizon_sec = self.get_parameter('projection_horizon_sec').value
        self._proj_dt = self.get_parameter('projection_dt').value
        self._proj_min_speed = self.get_parameter('projection_min_speed').value

        self._obs_soft = self.get_parameter('obs_soft_threshold').value
        self._obs_hard = self.get_parameter('obs_hard_threshold').value
        self._lane_soft = self.get_parameter('lane_soft_threshold').value
        self._lane_hard = self.get_parameter('lane_hard_threshold').value
        self._total_soft = self.get_parameter('total_soft_threshold').value
        self._total_hard = self.get_parameter('total_hard_threshold').value

        self._wheelbase = self.get_parameter('wheelbase').value

        self._latest_teleop = None     # AckermannDriveStamped
        self._latest_odom = None       # Odometry
        self._latest_obs = None        # MarkerArray

        self._ref_path = None          # RefPath built from /Routing/Path
        self._obstacle_dict = {}       # {id: (n,3) vertices}

        proj_T = max(2, int(round(self._proj_horizon_sec / self._proj_dt)))
        self._projector = ForwardProjector(
            wheelbase=self._wheelbase,
            dt=self._proj_dt,
            horizon=proj_T,
        )

        pkg_share = get_package_share_directory('racecar_ece346')
        ilqr_cfg_path = os.path.join(pkg_share, 'config', 'task2_ilqr.yaml')
        self._cost_eval = CostEvaluator(config_path=ilqr_cfg_path)

        self._ilqr = ILQR(logger=self.get_logger(), config_file=ilqr_cfg_path)

        # ---- TODO(Task 1.3): create subscribers ----
        self.create_subscription(
            AckermannDriveStamped, teleop_topic, self._teleop_cb, 1)
        self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 1)
        self.create_subscription(
            MarkerArray, obs_topic, self._obs_cb, 1)
        self.create_subscription(
            PathMsg, path_topic, self._path_cb, 10)

        # ---- TODO(Task 1.4): create the publisher ----
        self._drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 1)

        # ---- TODO(Task 1.5): create a timer at publish_rate Hz ----
        self.create_timer(1.0 / publish_rate, self._publish_filtered)

        self.get_logger().info(
            f"safety_filter_node ready: {teleop_topic} + {odom_topic} "
            f"+ {obs_topic} -> {drive_topic}"
        )

    # ---- TODO(Task 1.6): implement callbacks ----

    def _teleop_cb(self, msg):
        self._latest_teleop = msg

    def _odom_cb(self, msg):
        self._latest_odom = msg

    def _obs_cb(self, msg):
        self._latest_obs = msg
        self._obstacle_dict.clear()
        for marker in msg.markers:
            obs_id, verts = get_obstacle_vertices(marker)
            self._obstacle_dict[obs_id] = verts

    def _path_cb(self, msg):
        """Build a RefPath from the routing nav_msgs/Path message."""
        x_pts, y_pts = [], []
        width_L, width_R, speed_limit = [], [], []
        for wp in msg.poses:
            x_pts.append(wp.pose.position.x)
            y_pts.append(wp.pose.position.y)
            width_L.append(wp.pose.orientation.x)
            width_R.append(wp.pose.orientation.y)
            speed_limit.append(wp.pose.orientation.z)

        if len(x_pts) < 4:
            return

        centerline = np.array([x_pts, y_pts])
        try:
            self._ref_path = RefPath(
                centerline, width_L, width_R, speed_limit, loop=False)
            self.get_logger().info('Safety filter: reference path received.')
        except Exception as e:
            self.get_logger().warn(f'Invalid path: {e}')

    # ---- TODO(Task 1.7): the timer callback ----
    # Should:
    #   - return early if no teleop has arrived yet (self._latest_teleop is None)
    #   - call self.safety_filter(teleop=..., odom=..., obstacles=...)
    #   - if the returned command is not None, update its header.stamp to now
    #     and publish it on /drive
    #
    def _publish_filtered(self):
        if self._latest_teleop is None:
            return

        cmd = self.safety_filter(
            teleop=self._latest_teleop,
            odom=self._latest_odom,
            obstacles=self._latest_obs,
        )

        if cmd is not None:
            cmd.header.stamp = self.get_clock().now().to_msg()
            self._drive_pub.publish(cmd)

    # =========================================================================
    # TASK 2 — Safety filter implementation
    # =========================================================================
    #
    # Start as a passthrough, then add real safety logic.
    #
    # You are free to pick any approach (or your own). Add helper methods,
    # extra parameters, even a sub-folder of modules — this skeleton will
    # get out of the way.
    # =========================================================================

    def _extract_state(self, odom):
        """Build bicycle state [x, y, v, psi, delta] from Odometry."""
        o = odom.pose.pose.orientation
        psi = yaw_from_quat(o.x, o.y, o.z, o.w)
        v = odom.twist.twist.linear.x
        yaw_rate = odom.twist.twist.angular.z

        if abs(v) > 0.05:
            delta = math.atan2(yaw_rate * self._wheelbase, v)
        else:
            delta = 0.0
        delta = np.clip(delta, -0.35, 0.35)

        return np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            v,
            psi,
            delta,
        ])

    def _run_ilqr_override(self, state):
        """
        Run a lightweight ILQR plan from the current state to get a safe
        steering correction. Returns (speed, steering_angle) or None.
        """
        if self._ref_path is None:
            return None

        self._ilqr.update_ref_path(self._ref_path)

        obs_list = list(self._obstacle_dict.values())
        self._ilqr.update_obstacles(obs_list)

        try:
            plan = self._ilqr.plan(state)
        except Exception as e:
            self.get_logger().warn(f'ILQR plan failed: {e}')
            return None

        if plan.get('status', -1) == -1:
            return None

        traj = plan['trajectory']
        controls = plan['controls']

        accel_cmd = controls[0, 0]
        safe_speed = max(0.0, state[2] + accel_cmd * self._proj_dt)

        delta_next = traj[4, 1] if traj.shape[1] > 1 else state[4]
        return safe_speed, delta_next

    def safety_filter(self, teleop, odom, obstacles):
        """
        Args
        ----
        teleop : ackermann_msgs.msg.AckermannDriveStamped   (or None)
            Human's desired command. Useful fields:
                teleop.drive.speed           float   m/s     target forward speed
                teleop.drive.steering_angle  float   rad     target steering angle
                teleop.drive.acceleration    float   m/s^2   usually 0 from the joy
                teleop.header.stamp          Time            when the command was issued

        odom : nav_msgs.msg.Odometry                        (or None)
            Vehicle state. Useful fields:
                odom.pose.pose.position.x       float   m       map-frame x
                odom.pose.pose.position.y       float   m       map-frame y
                odom.pose.pose.position.z       float   m       usually 0
                odom.pose.pose.orientation      Quaternion (.x .y .z .w)
                    → use yaw_from_quat(..) above to get heading in rad
                odom.twist.twist.linear.x       float   m/s     forward velocity
                odom.twist.twist.angular.z      float   rad/s   yaw rate

        obstacles : visualization_msgs.msg.MarkerArray      (or None)
            Static obstacles (cubes). Useful fields:
                obstacles.markers               list[Marker]
                for m in obstacles.markers:
                    m.pose.position.x / .y / .z float   m       obstacle center
                    m.scale.x / .y / .z         float   m       cube size (x=y=z typically)
                    m.id                        int             obstacle id
                    m.ns                        str             namespace

        Returns
        -------
        ackermann_msgs.msg.AckermannDriveStamped
            The command to publish on /drive. Set `.drive.speed` (m/s) and
            `.drive.steering_angle` (rad). Header.stamp is overwritten for you.
            Return None to skip publishing this tick.
        """
        # ---- TODO(Task 2): replace this passthrough ----

        if odom is None:
            return teleop

        state = self._extract_state(odom)
        target_speed = teleop.drive.speed
        target_steer = teleop.drive.steering_angle

        trajectory, controls = self._projector.project(
            state, target_speed, target_steer,
            min_speed=self._proj_min_speed,
        )

        costs = self._cost_eval.evaluate(
            trajectory, controls, self._ref_path, self._obstacle_dict)

        obs_cost = costs['obs_cost']
        lane_cost = costs['lane_cost']
        total_cost = costs['total_cost']

        is_hard = (obs_cost > self._obs_hard or
                   lane_cost > self._lane_hard or
                   total_cost > self._total_hard)

        is_soft = (obs_cost > self._obs_soft or
                   lane_cost > self._lane_soft or
                   total_cost > self._total_soft)

        out = AckermannDriveStamped()
        out.header = teleop.header

        if is_hard:
            out.drive.speed = 0.0
            override = self._run_ilqr_override(state)
            if override is not None:
                out.drive.speed = 0.0
                out.drive.steering_angle = override[1]
            else:
                out.drive.steering_angle = 0.0
            return out

        if is_soft:
            override = self._run_ilqr_override(state)
            if override is not None:
                out.drive.speed = min(abs(target_speed), override[0])
                out.drive.steering_angle = override[1]
            else:
                speed_scale = 0.5
                out.drive.speed = target_speed * speed_scale
                out.drive.steering_angle = target_steer
            return out

        out.drive.speed = target_speed
        out.drive.steering_angle = target_steer
        return out


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
