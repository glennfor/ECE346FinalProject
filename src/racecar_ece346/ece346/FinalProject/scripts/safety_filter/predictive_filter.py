"""
Predictive safety filter with dual ILQR roles:

    monitor ILQR (higher costs): evaluates whether applying the human command
        for one ROS tick lands in an unsafe state.
    planner ILQR (nominal costs): produces the safe fallback command when the
        monitor flags unsafe.

Decision policy:
    1) If monitor cost is below threshold -> pass human command.
    2) If monitor cost is unsafe -> publish planner ILQR first control.
    3) If planner is also unsafe/invalid -> full brake fallback.

This keeps the monitor conservative while still using a less aggressive planner
for practical control when intervention is required.
"""

from typing import Optional, Tuple

import numpy as np


class PredictiveSafetyFilter:

    def __init__(
        self,
        planner_ilqr,
        monitor_ilqr,
        logger=None,
        monitor_max_allowed_cost: float = 3000.0,
        planner_max_allowed_cost: float = 4000.0,
        min_lane_margin: float = 0.03,
        kp_accel: float = 5.0,
        kp_steer: float = 6.0,
        delta_max: float = 0.35,
        accel_min: Optional[float] = None,
        accel_max: Optional[float] = None,
        omega_min: Optional[float] = None,
        omega_max: Optional[float] = None,
    ):
        """
        Args:
            planner_ilqr: ILQR instance used to generate fallback controls.
            monitor_ilqr: ILQR instance with higher costs for safety checking.
            logger: optional ROS logger.
            monitor_max_allowed_cost: maximum plan cost tolerated by the
                monitor. Above this is unsafe.
            planner_max_allowed_cost: maximum plan cost tolerated by the
                planner override. Above this is unsafe, so brake fallback.
            min_lane_margin: minimum lane margin in meters between the truck and the lane boundary
                along a planned path. geometric hard check different from smooth lane cost
            kp_accel, kp_steer: P-gains used to convert the human's
                (target_speed, target_steer) into bicycle controls
                (accel, omega) for the one-step forward simulation. Match
                the constants used in ForwardProjector.
            delta_max: steering clip used when mapping the fallback's first
                ILQR control back into a steering angle command.
            accel_min, accel_max, omega_min, omega_max: optional explicit
                control caps from node/yaml. If any is None, that bound falls
                back to planner ILQR ctrl_limits.
        """
        self._planner_ilqr = planner_ilqr
        self._monitor_ilqr = monitor_ilqr
        self._logger = logger
        self._monitor_max_allowed_cost = float(monitor_max_allowed_cost)
        self._planner_max_allowed_cost = float(planner_max_allowed_cost)
        self._min_lane_margin = float(min_lane_margin)
        self._kp_accel = float(kp_accel)
        self._kp_steer = float(kp_steer)
        self._delta_max = float(delta_max)
        self._accel_min = accel_min
        self._accel_max = accel_max
        self._omega_min = omega_min
        self._omega_max = omega_max

        self._u_warm_monitor = np.zeros((monitor_ilqr.dim_u, monitor_ilqr.T))
        self._u_warm_planner = np.zeros((planner_ilqr.dim_u, planner_ilqr.T))

        self._last_safe_steer = 0.0
        self._has_path = False
        self._has_obstacles = False
        self._ref_path = None

    def update_ref_path(self, ref_path) -> None:
        self._ref_path = ref_path
        self._planner_ilqr.update_ref_path(ref_path)
        self._monitor_ilqr.update_ref_path(ref_path)
        self._has_path = ref_path is not None
        self._u_warm_monitor.fill(0.0)
        self._u_warm_planner.fill(0.0)

    def update_obstacles(self, obstacle_dict: dict) -> None:
        """Push current obstacles into the shared ILQR.

        `obstacle_dict` is the {id: (n,3) vertices} mapping cached by the
        node from /Obstacles/Static.
        """
        obs_list = list(obstacle_dict.values()) if obstacle_dict else []
        self._planner_ilqr.update_obstacles(obs_list)
        self._monitor_ilqr.update_obstacles(obs_list)
        self._has_obstacles = len(obs_list) > 0

    def filter(
        self,
        state: np.ndarray,
        human_speed: float,
        human_steer: float,
        dt_step: float,
    ) -> Tuple[float, float, str]:
        """
        Returns (safe_speed, safe_steer, info) where info is one of
        'pass'      — human command accepted (monitor safe)
        'override'  — planner first control applied (monitor unsafe)
        'brake'     — monitor unsafe and planner unsafe; full brake
        'no_path'   — no ref path yet; passthrough human command
        """
        if not self._has_path:
            return float(human_speed), float(human_steer), 'no_path'

        accel_h, omega_h = self._teleop_command_to_controls(
            state, human_speed, human_steer)
        state_after = self._sim_forward(state, accel_h, omega_h, dt_step)

        monitor_plan = self._safe_plan(
            self._monitor_ilqr, state_after, self._u_warm_monitor)
        planner_plan = self._safe_plan(
            self._planner_ilqr, state, self._u_warm_planner)

        monitor_is_unsafe = self._is_unsafe(
            monitor_plan, self._monitor_max_allowed_cost)
        planner_is_unsafe = self._is_unsafe(
            planner_plan, self._planner_max_allowed_cost)

        if monitor_plan is not None and 'controls' in monitor_plan:
            self._u_warm_monitor = self._shift_controls(monitor_plan['controls'])
        if planner_plan is not None and 'controls' in planner_plan:
            self._u_warm_planner = self._shift_controls(planner_plan['controls'])

        if not monitor_is_unsafe:
            self._last_safe_steer = float(human_steer)
            return float(human_speed), float(human_steer), 'pass'

        if not planner_is_unsafe:
            safe_speed, safe_steer = self._first_control_to_cmd(
                planner_plan, state, dt_step)
            self._last_safe_steer = safe_steer
            return safe_speed, safe_steer, 'override'

        if self._logger is not None:
            self._logger.warn(
                'PredictiveSafetyFilter: monitor+planner unsafe — braking.')
        return 0.0, self._last_safe_steer, 'brake'

    def _teleop_command_to_controls(
        self, state: np.ndarray, target_speed: float, target_steer: float
    ) -> Tuple[float, float]:
        """Map (target_speed, target_steer) -> (accel, omega) via P-control.

        Simulated step matches the dynamics the human would experience under the existing low-level controllers.
        """
        v_cur = float(state[2])
        delta_cur = float(state[4])

        ctrl_lim = self._planner_ilqr.dyn.ctrl_limits
        a_min, a_max = float(ctrl_lim[0, 0]), float(ctrl_lim[0, 1])
        o_min, o_max = float(ctrl_lim[1, 0]), float(ctrl_lim[1, 1])

        accel = float(np.clip(self._kp_accel * (target_speed - v_cur), a_min, a_max))
        omega = float(np.clip(self._kp_steer * (target_steer - delta_cur), o_min, o_max))
        return accel, omega

    def _sim_forward(
        self, state: np.ndarray, accel: float, omega: float, dt_step: float
    ) -> np.ndarray:
        """Step the bicycle dynamics forward by dt_step using ILQR's model.

        Sub-steps at ilqr.dt to keep RK4 accuracy and to reuse the exact
        integrator the planner trusts.
        """
        ilqr_dt = float(self._planner_ilqr.dt)
        n_sub = max(1, int(round(dt_step / ilqr_dt)))
        accel, omega = self._clip_controls(accel, omega)
        u = np.array([accel, omega])
        x = np.asarray(state, dtype=float).copy()
        for _ in range(n_sub):
            x, _ = self._planner_ilqr.dyn.integrate_forward_np(x, u)
            x = np.asarray(x)
        return x

    def _safe_plan(self, ilqr, init_state: np.ndarray, warm_controls: np.ndarray) -> Optional[dict]:
        try:
            return ilqr.plan(init_state, controls=warm_controls.copy())
        except Exception as e:
            if self._logger is not None:
                self._logger.warn(f'PredictiveSafetyFilter: ILQR plan failed: {e}')
            return None

    def _is_unsafe(self, plan_result: Optional[dict], max_allowed_cost: float) -> bool:
        """Unsafe if invalid status/plan or if plan cost exceeds max allowed."""
        """Unsafe if invalid, high-cost, or geometrically outside the lane."""
        if plan_result is None:
            return True
        if plan_result.get('status', -1) == -1:
            return True

        traj = plan_result.get('trajectory')
        if traj is None:
            return True
        if self._violates_lane_margin(np.asarray(traj)):
            return True

        total_cost = plan_result.get('J')
        if total_cost is None:
            return True
        if not np.isfinite(total_cost):
            return True
        return bool(float(total_cost) >= float(max_allowed_cost))

    def _violates_lane_margin(self, trajectory: np.ndarray) -> bool:
        """Hard lane-boundary check for the whole vehicle body.

        The ILQR lane cost is smooth, so it may still produce a "recoverable"
        plan whose first state is already outside the lane but whose total cost
        is below threshold. The safety filter needs a hard invariant: do not
        accept the human command if the one-step future state or recovery plan
        leaves too little room to either boundary.
        """
        if self._ref_path is None:
            return False

        try:
            refs = self._ref_path.get_reference(trajectory[:2, :])
        except Exception as e:
            if self._logger is not None:
                self._logger.warn(
                    f'PredictiveSafetyFilter: lane check failed: {e}')
            return True

        closest_x = refs[0, :]
        closest_y = refs[1, :]
        slope = refs[2, :]
        width_right = refs[5, :]
        width_left = refs[6, :]

        dx = trajectory[0, :] - closest_x
        dy = trajectory[1, :] - closest_y
        path_dev = np.sin(slope) * dx - np.cos(slope) * dy

        vehicle_half_width = float(self._planner_ilqr.config.width) / 2.0
        right_margin = width_right - vehicle_half_width - path_dev
        left_margin = width_left - vehicle_half_width + path_dev
        min_margin = min(float(np.min(right_margin)), float(np.min(left_margin)))
        return min_margin < self._min_lane_margin

    def _first_control_to_cmd(
        self, plan: dict, state: np.ndarray, dt_step: float
    ) -> Tuple[float, float]:
        # """Apply ILQR's first stage control over one ROS tick."""
        # controls = np.asarray(plan['controls'])
        # accel_cmd = float(controls[0, 0])
        # omega_cmd = float(controls[1, 0])
        # accel_cmd, omega_cmd = self._clip_controls(accel_cmd, omega_cmd)
        # safe_speed = max(0.0, float(state[2]) + accel_cmd * dt_step)
        # safe_steer = float(np.clip(
        #     float(state[4]) + omega_cmd * dt_step,
        #     -self._delta_max, self._delta_max,
        # ))
        # return safe_speed, safe_steer

        """Map ILQR's next planned state to a speed/steering command.

        /drive.steering_angle is a steering position target, not a steering
        rate. Publishing state[4] + omega * dt_step under-commands the servo
        because dt_step is much smaller than the ILQR horizon step. Use the
        next planned wheel angle directly.
        """
        traj = np.asarray(plan['trajectory'])
        if traj.shape[1] > 1:
            safe_speed = max(0.0, float(traj[2, 1]))
            safe_steer = float(np.clip(
                traj[4, 1], -self._delta_max, self._delta_max))
        else:
            safe_speed = max(0.0, float(state[2]))
            safe_steer = float(np.clip(
                state[4], -self._delta_max, self._delta_max))
        return safe_speed, safe_steer

    def _clip_controls(self, accel: float, omega: float) -> Tuple[float, float]:
        """Clip controls to yaml limits (or ILQR defaults)."""
        ctrl_lim = self._planner_ilqr.dyn.ctrl_limits
        a_min = float(self._accel_min) if self._accel_min is not None else float(ctrl_lim[0, 0])
        a_max = float(self._accel_max) if self._accel_max is not None else float(ctrl_lim[0, 1])
        o_min = float(self._omega_min) if self._omega_min is not None else float(ctrl_lim[1, 0])
        o_max = float(self._omega_max) if self._omega_max is not None else float(ctrl_lim[1, 1])
        accel = float(np.clip(accel, a_min, a_max))
        omega = float(np.clip(omega, o_min, o_max))
        return accel, omega

    @staticmethod
    def _shift_controls(controls: np.ndarray) -> np.ndarray:
        """Shift a (dim_u, T) control sequence one step left for warm-start."""
        controls = np.asarray(controls)
        shifted = np.zeros_like(controls)
        if controls.shape[1] > 1:
            shifted[:, :-1] = controls[:, 1:]
            shifted[:, -1] = controls[:, -1]
        return shifted
