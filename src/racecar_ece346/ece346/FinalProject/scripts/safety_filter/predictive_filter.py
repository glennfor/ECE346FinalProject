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

import copy
import time
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
from ece346.FinalProject.ILQR_Example.cost.collision_checker.collision_checker import \
    CollisionChecker
from ece346.FinalProject.ILQR_Example.cost.collision_checker.obstacle import \
    Obstacle


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
            planner_replan_sec, planner_track_xy_thresh, planner_track_yaw_thresh:
                kept for call compatibility with safety_filter_node; values are
                IGNORED — use hardcoded self._planner_* assignments in __init__
                body below.

            # Original arg docs (when these kwargs drove behavior):
            # planner_replan_sec: minimum wall time between replans while overriding.
            # planner_track_xy_thresh: replan if |xy - nominal| exceeds this (m).
            # planner_track_yaw_thresh: replan if |yaw - nominal| exceeds this (rad).
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

        # --- Replan / nominal tracking: HARDcoded here (same defaults as old ROS params). ---
        # Node still passes planner_replan_sec / planner_track_* but they are ignored.
        self._planner_replan_sec = 0.15  # s; min host-monotonic interval between replans while overriding
        self._planner_track_xy_thresh = 0.35  # m; replan if |xy - x_bar| exceeds this at current stage k
        self._planner_track_yaw_thresh = 0.5  # rad; replan if |yaw - yaw_bar| exceeds this at stage k
        # prev impl: read from constructor / ROS:
        # self._planner_replan_sec = float(planner_replan_sec)
        # self._planner_track_xy_thresh = float(planner_track_xy_thresh)
        # self._planner_track_yaw_thresh = float(planner_track_yaw_thresh)
        _ignored_planner_tune_from_node = (
            planner_replan_sec, planner_track_xy_thresh, planner_track_yaw_thresh)

        self._last_planner_replan_mono = 0.0  # time.monotonic() stamp; Lab3 used ROS t in state[-1]

        self._u_warm_monitor = np.zeros((monitor_ilqr.dim_u, monitor_ilqr.T))
        self._u_warm_planner = np.zeros((planner_ilqr.dim_u, planner_ilqr.T))

        self._last_safe_steer = 0.0
        self._has_path = False
        self._has_obstacles = False
        self._ref_path = None
        self._obstacle_vertices = []
        self._obstacles = []  # list of Obstacle objects for collision checker
        self._brake_distance = 1.0  # meters — brake if min distance < this

        collision_cfg = SimpleNamespace(
            width=0.22, length=0.40, wheelbase=0.324, T=1)
        self._collision_checker = CollisionChecker(collision_cfg)

        # Cached planner plan state: when overriding, we consume this trajectory
        # step-by-step instead of replanning every tick.
        self._active_plan = None       # dict with 'trajectory' and 'controls'
        self._plan_step = 0            # current index into the cached plan
        self._is_overriding = False    # True while consuming the planner plan

        self._on_planner_plan = None
        self._on_monitor_plan = None

    def set_plan_callbacks(self, on_planner_plan=None, on_monitor_plan=None):
        """Register callbacks called with the (5, T) trajectory each tick.

        on_planner_plan(trajectory_np)  — called after the planner solve.
        on_monitor_plan(trajectory_np)  — called after the monitor solve.
        """
        self._on_planner_plan = on_planner_plan
        self._on_monitor_plan = on_monitor_plan

    def update_ref_path(self, ref_path) -> None:
        self._ref_path = copy.deepcopy(ref_path)
        self._planner_ilqr.update_ref_path(ref_path)
        self._monitor_ilqr.update_ref_path(ref_path)
        self._has_path = ref_path is not None
        self._u_warm_monitor.fill(0.0)
        self._u_warm_planner.fill(0.0)
        self._exit_override()

    def update_obstacles(self, obstacle_dict: dict) -> None:
        """Push current obstacles into the shared ILQR.

        `obstacle_dict` is the {id: (n,3) vertices} mapping cached by the
        node from /Obstacles/Static.
        """
        obs_list = list(obstacle_dict.values()) if obstacle_dict else []
        self._obstacle_vertices = obs_list
        self._obstacles = [Obstacle(verts) for verts in obs_list]
        self._planner_ilqr.update_obstacles(obs_list)
        self._monitor_ilqr.update_obstacles(obs_list)
        self._has_obstacles = len(obs_list) > 0

    def _obstacle_too_close(self, state: np.ndarray) -> bool:
        """Return True if any obstacle's signed distance to the vehicle body < brake_distance."""
        if not self._obstacles:
            return False
        state_col = state.reshape(5, 1)
        try:
            refs = self._collision_checker.check_collisions(state_col, self._obstacles)
        except Exception:
            return False
        if refs is None:
            return False
        min_dist = float(np.min(refs[:, 4, :]))
        return min_dist < self._brake_distance

    def _generate_plan(self, state: np.ndarray) -> Optional[dict]:
        """Generate a fresh planner plan from current state, warm-starting if possible."""
        warm = self._u_warm_planner if np.any(self._u_warm_planner) else None
        plan = self._safe_plan(self._planner_ilqr, state, warm)
        if plan is None:
            if self._logger is not None:
                self._logger.warn('PredictiveSafetyFilter: planner returned None')
            return None
        if plan.get('status', -1) in (-1, 2):
            if self._logger is not None:
                self._logger.warn(
                    f'PredictiveSafetyFilter: planner status={plan.get("status")}')
            return None
        if 'trajectory' not in plan or 'controls' not in plan:
            return None
        return plan

    def _enter_override(self, state: np.ndarray) -> bool:
        """Plan once when first entering override mode. Returns True if plan succeeded."""
        plan = self._generate_plan(np.asarray(state, dtype=float).ravel()[:5])
        if plan is None:
            return False
        self._active_plan = plan
        self._plan_step = 0
        self._is_overriding = True
        self._u_warm_planner = np.asarray(plan['controls']).copy()
        self._last_planner_replan_mono = time.monotonic()
        return True

    def _maybe_replan_planner(self, state: np.ndarray) -> None:
        """Receding-horizon style replan from measured state (time + tracking error)."""
        if self._active_plan is None or not self._is_overriding:
            return
        now = time.monotonic()
        traj = np.asarray(self._active_plan['trajectory'])
        T = traj.shape[1]
        k = min(self._plan_step, T - 1)
        timed = (now - self._last_planner_replan_mono) >= self._planner_replan_sec
        xy_err = float(np.linalg.norm(state[:2] - traj[:2, k]))
        yaw_err = float(
            abs((state[3] - traj[3, k] + np.pi) % (2.0 * np.pi) - np.pi))
        off_nominal = (
            xy_err > self._planner_track_xy_thresh
            or yaw_err > self._planner_track_yaw_thresh)
        if not timed and not off_nominal:
            return
        warm = self._shift_controls(np.asarray(self._active_plan['controls']))
        self._u_warm_planner = warm
        plan = self._generate_plan(np.asarray(state, dtype=float).ravel()[:5])
        if plan is None:
            self._active_plan = None
            self._plan_step = 0
            return
        self._active_plan = plan
        self._plan_step = 0
        self._u_warm_planner = np.asarray(plan['controls']).copy()
        self._last_planner_replan_mono = now

    def _planner_override_command(
            self, state: np.ndarray) -> Optional[Tuple[float, float]]:
        """Lab3-style local LQR: u = u_bar + k + K(x - x_bar), then one ILQR dynamics step."""
        if self._active_plan is None:
            return None
        x = np.asarray(state, dtype=float).ravel()[:5]
        traj = np.asarray(self._active_plan['trajectory'])
        ctrl = np.asarray(self._active_plan['controls'])
        T = traj.shape[1]
        k = min(self._plan_step, T - 1)
        x_bar = traj[:, k]
        u_bar = ctrl[:, k]

        K = self._active_plan.get('K_closed_loop')
        k_open = self._active_plan.get('k_open_loop')
        x_diff = x - x_bar
        x_diff[3] = (x_diff[3] + np.pi) % (2.0 * np.pi) - np.pi

        if K is not None and getattr(K, 'size', 0) > 0:
            Kk = K[:, :, k]
            kk = (k_open[:, k] if k_open is not None
                  else np.zeros(self._planner_ilqr.dim_u))
            u = np.asarray(u_bar + kk + Kk @ x_diff, dtype=float).ravel()
        else:
            u = np.asarray(u_bar, dtype=float).ravel()

        ua, uo = self._clip_controls(float(u[0]), float(u[1]))
        u_cmd = np.array([ua, uo], dtype=float)
        try:
            x_next, _ = self._planner_ilqr.dyn.integrate_forward_np(x, u_cmd)
        except Exception:
            return None
        x_next = np.asarray(x_next, dtype=float).ravel()
        safe_speed = max(0.0, float(x_next[2]))
        safe_steer = float(np.clip(x_next[4], -self._delta_max, self._delta_max))

        self._plan_step += 1
        if self._plan_step >= T - 1:
            warm = self._shift_controls(ctrl)
            self._u_warm_planner = warm
            plan = self._generate_plan(x)
            if plan is None:
                self._active_plan = None
                self._is_overriding = False
                self._plan_step = 0
                return None
            self._active_plan = plan
            self._plan_step = 0
            self._u_warm_planner = np.asarray(plan['controls']).copy()
            self._last_planner_replan_mono = time.monotonic()
        return safe_speed, safe_steer

    def _exit_override(self):
        """Release planner plan when human is safe again."""
        self._is_overriding = False
        self._active_plan = None
        self._plan_step = 0

    def filter(
        self,
        state: np.ndarray,
        human_speed: float,
        human_steer: float,
        dt_step: float,
    ) -> Tuple[float, float, str]:
        """
        Returns (safe_speed, safe_steer, info) where info is one of
        'pass'            — human command accepted (monitor safe)
        'override'        — consuming cached planner trajectory
        'override_replan' — replanned because trajectory was exhausted
        'brake'           — override failed; full brake
        'no_path'         — no ref path yet; passthrough human command
        'proximity_brake' — obstacle within brake_distance; full stop
        """
        if self._obstacle_too_close(state):
            if self._logger is not None:
                self._logger.warn(
                    f'PredictiveSafetyFilter: obstacle within {self._brake_distance}m — braking')
            self._exit_override()
            return 0.0, 0.0, 'proximity_brake'
        if not self._has_path:
            return float(human_speed), float(human_steer), 'no_path'

        # --- Monitor: check if human command is safe ---
        accel_h, omega_h = self._teleop_command_to_controls(
            state, human_speed, human_steer)
        state_after = self._sim_forward(state, accel_h, omega_h, dt_step)

        monitor_plan = self._safe_plan(self._monitor_ilqr, state_after, None)
        monitor_is_unsafe, monitor_reason = self._is_unsafe(
            monitor_plan, self._monitor_max_allowed_cost)

        if monitor_plan is not None and 'controls' in monitor_plan:
            self._u_warm_monitor = self._shift_controls(monitor_plan['controls'])

        if self._on_monitor_plan and monitor_plan and 'trajectory' in monitor_plan:
            self._on_monitor_plan(np.asarray(monitor_plan['trajectory']))

        # if self._logger is not None:
        #     self._logger.warn(
        #         f'PredictiveSafetyFilter:. '
        #         f'monitor[{monitor_reason}] '
        #         # f'| planner[{planner_reason}] | '
        #         f'v={float(state[2]):.2f} delta={float(state[4]):.3f} '
        #         f'human(speed={float(human_speed):.2f},steer={float(human_steer):.3f})'
        #     )
        # add for braking
        # if self._logger is not None:
        #     override_str = f' step={self._plan_step}' if self._is_overriding else ''
        #     self._logger.warn(
        #         f'PredictiveSafetyFilter: '
        #         f'monitor[{monitor_reason}]{override_str} '
        #         f'v={float(state[2]):.2f} delta={float(state[4]):.3f} '
        #         f'human(speed={float(human_speed):.2f},steer={float(human_steer):.3f})'
        #     )

        # --- Decision ---
        if not monitor_is_unsafe:
            # Human is safe — release override if active
            if self._is_overriding:
                self._exit_override()
            self._last_safe_steer = float(human_steer)
            return float(human_speed), float(human_steer), 'pass'
        
        # if self._logger is not None:
        #     self._logger.warn(
        #         f'PredictiveSafetyFilter:. '
        #         f'monitor[{monitor_reason}]'
        #         # f' | planner[{planner_reason}] | '
        #         f'v={float(state[2]):.2f} delta={float(state[4]):.3f} '
        #         f'human(speed={float(human_speed):.2f},steer={float(human_steer):.3f})'
        #     )
        

        # return 0.0, float(human_steer), 'Stopped'

        # Monitor says unsafe — cached planner + LQR tracking + periodic replanning
        if (not self._is_overriding) or (self._active_plan is None):
            if not self._enter_override(state):
                return 0.0, self._last_safe_steer, 'brake'

        self._maybe_replan_planner(state)
        if self._active_plan is None:
            self._is_overriding = False
            return 0.0, self._last_safe_steer, 'brake'

        result = self._planner_override_command(state)

        if self._on_planner_plan and self._active_plan and 'trajectory' in self._active_plan:
            self._on_planner_plan(np.asarray(self._active_plan['trajectory']))

        if result is None:
            self._exit_override()
            return 0.0, self._last_safe_steer, 'brake'

        safe_speed, safe_steer = result
        self._last_safe_steer = safe_steer
        return safe_speed, safe_steer, 'override'

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
        # ilqr_dt = float(self._planner_ilqr.dt)
        # n_sub = max(1, int(round(dt_step / ilqr_dt)))
        # accel, omega = self._clip_controls(accel, omega)
        # u = np.array([accel, omega])
        # x = np.asarray(state, dtype=float).copy()
        # for _ in range(n_sub):
        #     x, _ = self._planner_ilqr.dyn.integrate_forward_np(x, u)
        #     x = np.asarray(x)
        # return x
        # return self._rk4_step(state, accel, omega, dt_step)
        return self.dyn_steps(state, [accel, omega], dt_step, 5)
    
    def dyn_steps(self, x, u, dt, times = 1):
        for _ in range(times):
            dx = np.array([x[2]*np.cos(x[3]),
                        x[2]*np.sin(x[3]),
                        u[0],
                        x[2]*np.tan(u[1]*1.1)/0.257,
                        0
                        ])
            x_new = x + dx*dt
            x_new[2] = max(0, x_new[2]) # do not allow negative velocity
            x_new[3] = np.mod(x_new[3] + np.pi, 2 * np.pi) - np.pi
            x_new[-1] = u[1]
            x = x_new
        return x_new
    
    def _deriv(self, state: np.ndarray, accel: float, omega: float) -> np.ndarray:
        x, y, v, psi, delta = state
        return np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            accel,
            v * np.tan(delta) /  0.324, # self.wheelbase,
            omega,
        ])

    def _rk4_step(self, state: np.ndarray, accel: float, omega: float, dt) -> np.ndarray:
        k1 = self._deriv(state, accel, omega)
        k2 = self._deriv(state + k1 * dt / 2, accel, omega)
        k3 = self._deriv(state + k2 * dt / 2, accel, omega)
        k4 = self._deriv(state + k3 * dt, accel, omega)
        state_next = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        state_next[2] = np.clip(state_next[2], 0.0, 0.4)
        state_next[3] = np.atan2(np.sin(state_next[3]), np.cos(state_next[3]))
        state_next[4] = np.clip(state_next[4], -0.35, 35)
        return state_next

    def _safe_plan(self, ilqr, init_state: np.ndarray, warm_controls: Optional[np.ndarray] = None) -> Optional[dict]:
        try:
            controls = warm_controls.copy() if warm_controls is not None else None
            return ilqr.plan(init_state, controls=controls)
        except Exception as e:
            if self._logger is not None:
                self._logger.warn(f'PredictiveSafetyFilter: ILQR plan failed: {e}')
            return None

    def _is_unsafe(self, plan_result: Optional[dict], max_allowed_cost: float, skip_first_state: bool = False) -> Tuple[bool, str]:
        """Return (is_unsafe, reason). reason is a short string with values.

        Unsafe if invalid status/plan, geometrically outside the lane, or if
        plan cost exceeds max allowed.
        """
        if plan_result is None:
            return True, 'plan=None'
        if plan_result.get('status', -1) in (-1, 2):
            return True, f'status={plan_result.get("status")}'

        traj = plan_result.get('trajectory')
        if traj is None:
            return True, 'trajectory=None'
        
        # TEST CODE
        check_traj = np.asarray(traj)
        if skip_first_state and check_traj.shape[1] > 1:
            check_traj = check_traj[:, 1:]

        # END TEST #margin = ....(np.asarray(traj))

        margin = self._lane_min_margin(check_traj)
        total_cost = plan_result.get('J')
        cost_str = (f'{float(total_cost):.1f}'
                    if total_cost is not None and np.isfinite(total_cost)
                    else f'{total_cost}')

        # if margin is not None and margin < self._min_lane_margin:
        #     return True, (f'lane min_margin={margin:.3f}<{self._min_lane_margin:.3f}m '
        #                   f'(J={cost_str})')

        if total_cost is None or not np.isfinite(total_cost):
            return True, f'J={cost_str}'

        if float(total_cost) >= float(max_allowed_cost):
            return True, f'J={cost_str}>=max={float(max_allowed_cost):.1f}'

        return False, f'OK J={cost_str}'

    def _lane_min_margin(self, trajectory: np.ndarray) -> Optional[float]:
        """Min lane margin (m) along trajectory; None if no path.

        Negative means the vehicle body crosses the boundary by that amount.
        Returns -inf on lookup exception (treated as infinite violation).
        """
        if self._ref_path is None:
            return None

        try:
            refs = self._ref_path.get_reference(trajectory[:2, :])
        except Exception as e:
            if self._logger is not None:
                self._logger.warn(
                    f'PredictiveSafetyFilter: lane check failed: {e}')
            return float('-inf')

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
        return float(min(np.min(right_margin), np.min(left_margin)))

    def _violates_lane_margin(self, trajectory: np.ndarray) -> bool:
        """Hard lane-boundary check for the whole vehicle body.

        The ILQR lane cost is smooth, so it may still produce a "recoverable"
        plan whose first state is already outside the lane but whose total cost
        is below threshold. The safety filter needs a hard invariant: do not
        accept the human command if the one-step future state or recovery plan
        leaves too little room to either boundary.
        """
        margin = self._lane_min_margin(trajectory)
        if margin is None:
            return False
        return margin < self._min_lane_margin

    def _first_control_to_cmd(
        self, plan: dict, state: np.ndarray, dt_step: float
    ) -> Tuple[float, float]:
        """Apply ILQR's first stage control over one ROS tick."""
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
            # changed k for 1
            k = min(8, traj.shape[1] -1)
            safe_speed = max(0.0, float(traj[2, k]))
            safe_steer = float(np.clip(
                traj[4, k], -self._delta_max, self._delta_max))
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
