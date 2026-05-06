"""
Predictive (two-ILQR) safety filter.

A control input from the human driver is considered safe iff, after applying it
for one ROS tick, there still exists a feasible safe trajectory from the
resulting state. This is checked with two ILQR plans per tick:

    plan_a: from x_after_human  -> "is there a recovery plan if we let the
                                    human act for one tick?"
    plan_b: from x_current      -> the safe optimal plan from now; used as the
                                    fallback control when plan_a fails.

A plan is "feasible" when the planner produced a trajectory and the minimum
geometric distance to every obstacle stays above `min_clearance`. We compute
clearance with the same hppfcl collision checker the ILQR uses, so the check
is independent of the smooth barrier cost weights and is not fooled by ILQR
converging to a locally optimal but corner-cutting plan.

The single shared ILQR instance is reused for both plans. Each role keeps its
own warm-start nominal control buffer so successive plans converge fast.
"""

from typing import Optional, Tuple

import numpy as np


class PredictiveSafetyFilter:

    def __init__(
        self,
        ilqr,
        logger=None,
        min_clearance: float = 0.0,
        kp_accel: float = 5.0,
        kp_steer: float = 6.0,
        delta_max: float = 0.35,
    ):
        """
        Args:
            ilqr: shared ILQR instance (its dyn, cost, collision_checker,
                ref_path and obstacle_list are reused).
            logger: optional ROS logger.
            min_clearance: minimum required signed distance (m) between ego
                and any obstacle along the planned horizon for the plan to
                be considered feasible. 0.0 means "no penetration"; raise to
                e.g. 0.05 for a safety buffer.
            kp_accel, kp_steer: P-gains used to convert the human's
                (target_speed, target_steer) into bicycle controls
                (accel, omega) for the one-step forward simulation. Match
                the constants used in ForwardProjector.
            delta_max: steering clip used when mapping the fallback's first
                ILQR control back into a steering angle command.
        """
        self._ilqr = ilqr
        self._logger = logger
        self._min_clearance = float(min_clearance)
        self._kp_accel = float(kp_accel)
        self._kp_steer = float(kp_steer)
        self._delta_max = float(delta_max)

        self._u_warm_a = np.zeros((ilqr.dim_u, ilqr.T))
        self._u_warm_b = np.zeros((ilqr.dim_u, ilqr.T))

        self._last_safe_steer = 0.0
        self._has_path = False
        self._has_obstacles = False

    def update_ref_path(self, ref_path) -> None:
        self._ilqr.update_ref_path(ref_path)
        self._has_path = ref_path is not None
        self._u_warm_a.fill(0.0)
        self._u_warm_b.fill(0.0)

    def update_obstacles(self, obstacle_dict: dict) -> None:
        """Push current obstacles into the shared ILQR.

        `obstacle_dict` is the {id: (n,3) vertices} mapping cached by the
        node from /Obstacles/Static.
        """
        obs_list = list(obstacle_dict.values()) if obstacle_dict else []
        self._ilqr.update_obstacles(obs_list)
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
        'pass'      — human command accepted (plan_a feasible)
        'override'  — fallback ILQR_B's first control applied
        'brake'     — both plans infeasible; full brake with last steer
        'no_path'   — no ref path yet; passthrough human command
        """
        if not self._has_path:
            return float(human_speed), float(human_steer), 'no_path'

        accel_h, omega_h = self._human_to_uvec(state, human_speed, human_steer)
        state_after = self._sim_forward(state, accel_h, omega_h, dt_step)

        plan_a = self._safe_plan(state_after, self._u_warm_a)
        plan_b = self._safe_plan(state, self._u_warm_b)

        feas_a = self._verify(plan_a)
        feas_b = self._verify(plan_b)

        if plan_a is not None and 'controls' in plan_a:
            self._u_warm_a = self._shift_controls(plan_a['controls'])
        if plan_b is not None and 'controls' in plan_b:
            self._u_warm_b = self._shift_controls(plan_b['controls'])

        if feas_a:
            self._last_safe_steer = float(human_steer)
            return float(human_speed), float(human_steer), 'pass'

        if feas_b:
            safe_speed, safe_steer = self._first_control_to_cmd(
                plan_b, state, dt_step)
            self._last_safe_steer = safe_steer
            return safe_speed, safe_steer, 'override'

        if self._logger is not None:
            self._logger.warn(
                'PredictiveSafetyFilter: both plans infeasible — braking.')
        return 0.0, self._last_safe_steer, 'brake'

    def _human_to_uvec(
        self, state: np.ndarray, target_speed: float, target_steer: float
    ) -> Tuple[float, float]:
        """Map (target_speed, target_steer) -> (accel, omega) via P-control.

        Mirrors ForwardProjector so the simulated step matches the dynamics
        the human would experience under the existing low-level controllers.
        """
        v_cur = float(state[2])
        delta_cur = float(state[4])

        ctrl_lim = self._ilqr.dyn.ctrl_limits
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
        ilqr_dt = float(self._ilqr.dt)
        n_sub = max(1, int(round(dt_step / ilqr_dt)))
        u = np.array([accel, omega])
        x = np.asarray(state, dtype=float).copy()
        for _ in range(n_sub):
            x, _ = self._ilqr.dyn.integrate_forward_np(x, u)
            x = np.asarray(x)
        return x

    def _safe_plan(self, init_state: np.ndarray, warm_controls: np.ndarray) -> Optional[dict]:
        try:
            return self._ilqr.plan(init_state, controls=warm_controls.copy())
        except Exception as e:
            if self._logger is not None:
                self._logger.warn(f'PredictiveSafetyFilter: ILQR plan failed: {e}')
            return None

    def _verify(self, plan_result: Optional[dict]) -> bool:
        """Trajectory-level feasibility check (status + min clearance)."""
        if plan_result is None:
            return False
        if plan_result.get('status', -1) == -1:
            return False
        traj = plan_result.get('trajectory')
        if traj is None:
            return False

        traj = np.asarray(traj)
        obs_list = self._ilqr.obstacle_list
        if not obs_list:
            return True

        prev_step = self._ilqr.collision_checker.step
        try:
            self._ilqr.collision_checker.step = traj.shape[1]
            obs_refs = self._ilqr.collision_checker.check_collisions(traj, obs_list)
        except Exception as e:
            if self._logger is not None:
                self._logger.warn(
                    f'PredictiveSafetyFilter: collision check failed: {e}')
            return False
        finally:
            self._ilqr.collision_checker.step = prev_step

        if obs_refs is None:
            return True

        distances = obs_refs[:, 4, :]
        if not np.all(np.isfinite(distances)):
            return False
        return bool(np.min(distances) >= self._min_clearance)

    def _first_control_to_cmd(
        self, plan: dict, state: np.ndarray, dt_step: float
    ) -> Tuple[float, float]:
        """Apply ILQR's first stage control over one ROS tick."""
        controls = np.asarray(plan['controls'])
        accel_cmd = float(controls[0, 0])
        omega_cmd = float(controls[1, 0])
        safe_speed = max(0.0, float(state[2]) + accel_cmd * dt_step)
        safe_steer = float(np.clip(
            float(state[4]) + omega_cmd * dt_step,
            -self._delta_max, self._delta_max,
        ))
        return safe_speed, safe_steer

    @staticmethod
    def _shift_controls(controls: np.ndarray) -> np.ndarray:
        """Shift a (dim_u, T) control sequence one step left for warm-start."""
        controls = np.asarray(controls)
        shifted = np.zeros_like(controls)
        if controls.shape[1] > 1:
            shifted[:, :-1] = controls[:, 1:]
            shifted[:, -1] = controls[:, -1]
        return shifted
