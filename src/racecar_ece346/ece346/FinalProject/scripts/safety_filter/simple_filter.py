"""
Simple safety filter: brake when unsafe, pass through when safe.

Decision logic:
    1. Forward-project the car's trajectory under the human command.
    2. Check if any projected state violates lane boundaries or is too
       close to an obstacle.
    3. If unsafe -> brake (speed = 0), keep human steering so they can correct.
    4. If safe -> pass through human command unchanged.

No ILQR override, no steering correction. The human stays in control of
steering at all times. The filter only removes speed when danger is detected.
"""

from typing import Tuple

import numpy as np


class SimpleSafetyFilter:

    def __init__(
        self,
        projector,
        logger=None,
        lane_margin: float = 0.05,
        obs_margin: float = 0.15,
        vehicle_half_width: float = 0.15,
        lookahead_steps: int = 8,
    ):
        self._projector = projector
        self._logger = logger
        self._lane_margin = float(lane_margin)
        self._obs_margin = float(obs_margin)
        self._vehicle_half_width = float(vehicle_half_width)
        self._lookahead_steps = int(lookahead_steps)

        self._ref_path = None
        self._has_path = False
        self._obstacle_list = []

    def update_ref_path(self, ref_path) -> None:
        self._ref_path = ref_path
        self._has_path = ref_path is not None

    def update_obstacles(self, obstacle_dict: dict) -> None:
        self._obstacle_list = list(obstacle_dict.values()) if obstacle_dict else []

    def filter(
        self,
        state: np.ndarray,
        human_speed: float,
        human_steer: float,
        **kwargs,
    ) -> Tuple[float, float, str]:
        """
        Returns (safe_speed, safe_steer, info) where info is one of:
            'pass'    - human command accepted (trajectory is safe)
            'brake'   - unsafe detected; speed zeroed, steering kept
            'no_path' - no ref path yet; passthrough
        """
        if not self._has_path:
            return float(human_speed), float(human_steer), 'no_path'

        trajectory, _ = self._projector.project(state, human_speed, human_steer)

        steps_to_check = min(self._lookahead_steps, trajectory.shape[1])
        traj_check = trajectory[:, :steps_to_check]

        lane_unsafe = self._check_lane_violation(traj_check)
        obs_unsafe = self._check_obstacle_violation(traj_check)

        if lane_unsafe or obs_unsafe:
            reason = []
            if lane_unsafe:
                reason.append('lane')
            if obs_unsafe:
                reason.append('obstacle')
            info = f'brake({"+".join(reason)})'

            if self._logger is not None:
                self._logger.warn(
                    f'SimpleBrakeFilter: {info} | v={state[2]:.2f} '
                    f'human(spd={human_speed:.2f}, str={human_steer:.3f})')

            return 0.0, float(human_steer), info

        return float(human_speed), float(human_steer), 'pass'

    def _check_lane_violation(self, trajectory: np.ndarray) -> bool:
        if self._ref_path is None:
            return False

        try:
            refs = self._ref_path.get_reference(trajectory[:2, :])
        except Exception:
            return True

        closest_x = refs[0, :]
        closest_y = refs[1, :]
        slope = refs[2, :]
        width_right = refs[5, :]
        width_left = refs[6, :]

        dx = trajectory[0, :] - closest_x
        dy = trajectory[1, :] - closest_y
        lateral_dev = np.sin(slope) * dx - np.cos(slope) * dy

        right_margin = width_right - self._vehicle_half_width - lateral_dev
        left_margin = width_left - self._vehicle_half_width + lateral_dev

        min_margin = float(min(np.min(right_margin), np.min(left_margin)))
        return min_margin < self._lane_margin

    def _check_obstacle_violation(self, trajectory: np.ndarray) -> bool:
        if not self._obstacle_list:
            return False

        for obs_verts in self._obstacle_list:
            verts = np.asarray(obs_verts)
            obs_center_x = np.mean(verts[:, 0])
            obs_center_y = np.mean(verts[:, 1])
            obs_radius = np.max(np.sqrt(
                (verts[:, 0] - obs_center_x) ** 2 +
                (verts[:, 1] - obs_center_y) ** 2
            ))

            dists = np.sqrt(
                (trajectory[0, :] - obs_center_x) ** 2 +
                (trajectory[1, :] - obs_center_y) ** 2
            )

            if np.any(dists < (obs_radius + self._obs_margin)):
                return True

        return False