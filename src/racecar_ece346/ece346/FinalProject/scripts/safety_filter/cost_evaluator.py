import os
import numpy as np

from ece346.FinalProject.ILQR_Example.config import Config
from ece346.FinalProject.ILQR_Example.cost.state_cost import StateCost
from ece346.FinalProject.ILQR_Example.cost.control_cost import ControlCost
from ece346.FinalProject.ILQR_Example.cost.obstacle_cost import ObstacleCost
from ece346.FinalProject.ILQR_Example.cost.collision_checker.collision_checker import CollisionChecker
from ece346.FinalProject.ILQR_Example.cost.collision_checker.obstacle import Obstacle


class CostEvaluator:
    """
    Evaluates three independent cost channels on a projected trajectory:
      - obs_cost:   obstacle collision cost only
      - lane_cost:  state cost (path dev + boundary + heading + speed)
      - total_cost: obs + state + control combined
    """

    def __init__(self, config_path: str = None):
        self.config = Config()
        if config_path and os.path.isfile(config_path):
            self.config.load_config(config_path)

        self.state_cost = StateCost(self.config)
        self.control_cost = ControlCost(self.config)
        self.obstacle_cost = ObstacleCost(self.config)
        self.collision_checker = CollisionChecker(self.config)

    def evaluate(self, trajectory: np.ndarray, controls: np.ndarray,
                 ref_path, obstacle_vertices: dict) -> dict:
        """
        Compute the three cost channels for a projected trajectory.

        Args:
            trajectory: (5, T) state trajectory
            controls: (2, T) control sequence
            ref_path: RefPath object (or None if no path available)
            obstacle_vertices: dict of {id: (n, 3) vertices} for each obstacle

        Returns:
            dict with keys 'obs_cost', 'lane_cost', 'total_cost'
        """
        T = trajectory.shape[1]
        result = {'obs_cost': 0.0, 'lane_cost': 0.0, 'total_cost': 0.0}

        # Ensure collision checker step count matches trajectory length
        self.collision_checker.step = T

        path_refs = None
        if ref_path is not None:
            try:
                path_refs = ref_path.get_reference(trajectory[:2, :])
            except Exception:
                path_refs = None

        obs_list = []
        for verts in obstacle_vertices.values():
            obs_list.append(Obstacle(verts))

        obs_refs = self.collision_checker.check_collisions(trajectory, obs_list) \
            if obs_list else None

        if obs_refs is not None:
            obs_cost = float(self.obstacle_cost.get_traj_cost(
                trajectory, controls, obs_refs))
            result['obs_cost'] = obs_cost

        if path_refs is not None:
            lane_cost = float(self.state_cost.get_traj_cost(
                trajectory, controls, path_refs))
            result['lane_cost'] = lane_cost

        ctrl_cost = 0.0
        if path_refs is not None:
            ctrl_cost = float(self.control_cost.get_traj_cost(
                trajectory, controls, path_refs))

        result['total_cost'] = result['obs_cost'] + result['lane_cost'] + ctrl_cost

        return result
