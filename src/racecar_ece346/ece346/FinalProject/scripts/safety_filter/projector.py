import numpy as np
import math


class ForwardProjector:
    """
    Forward-projects vehicle state under a constant Ackermann command
    using a kinematic bicycle model (RK4 integration).
    """

    def __init__(self, wheelbase: float, dt: float, horizon: int,
                 v_max: float = 5.0, delta_max: float = 0.35):
        self.wheelbase = wheelbase
        self.dt = dt
        self.horizon = horizon
        self.v_max = v_max
        self.delta_max = delta_max

    def _deriv(self, state: np.ndarray, accel: float, omega: float) -> np.ndarray:
        x, y, v, psi, delta = state
        return np.array([
            v * math.cos(psi),
            v * math.sin(psi),
            accel,
            v * math.tan(delta) / self.wheelbase,
            omega,
        ])

    def _rk4_step(self, state: np.ndarray, accel: float, omega: float) -> np.ndarray:
        k1 = self._deriv(state, accel, omega)
        k2 = self._deriv(state + k1 * self.dt / 2, accel, omega)
        k3 = self._deriv(state + k2 * self.dt / 2, accel, omega)
        k4 = self._deriv(state + k3 * self.dt, accel, omega)
        state_next = state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

        state_next[2] = np.clip(state_next[2], 0.0, self.v_max)
        state_next[3] = math.atan2(math.sin(state_next[3]), math.cos(state_next[3]))
        state_next[4] = np.clip(state_next[4], -self.delta_max, self.delta_max)
        return state_next

    def project(self, state: np.ndarray, target_speed: float,
                target_steer: float, min_speed: float = 0.5) -> tuple:
        """
        Roll out bicycle dynamics from `state` for self.horizon steps.

        The controls are computed to drive the vehicle toward the target
        speed and steering angle using simple proportional controllers.

        Args:
            state: [x, y, v, psi, delta] current bicycle state
            target_speed: desired forward speed (m/s)
            target_steer: desired steering angle (rad)
            min_speed: floor on the projection speed so we look ahead even
                       when the driver is nearly stopped

        Returns:
            trajectory: (5, T) state trajectory
            controls: (2, T) control sequence [accel, omega]
        """
        T = self.horizon
        trajectory = np.zeros((5, T))
        controls = np.zeros((2, T))
        trajectory[:, 0] = state.copy()

        kp_accel = 5.0
        kp_steer = 6.0

        proj_speed = max(abs(target_speed), min_speed)

        for t in range(T - 1):
            v_cur = trajectory[2, t]
            delta_cur = trajectory[4, t]

            accel = np.clip(kp_accel * (proj_speed - v_cur), -5.0, 5.0)
            omega = np.clip(kp_steer * (target_steer - delta_cur), -6.0, 6.0)

            controls[:, t] = [accel, omega]
            trajectory[:, t + 1] = self._rk4_step(trajectory[:, t], accel, omega)

        controls[:, -1] = controls[:, -2] if T > 1 else [0.0, 0.0]
        return trajectory, controls
