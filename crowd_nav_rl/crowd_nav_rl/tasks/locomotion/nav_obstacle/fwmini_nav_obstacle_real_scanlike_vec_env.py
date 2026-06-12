"""Scan-like observation variant of the FW-mini obstacle navigation env.

This class deliberately reuses the existing obstacle env dynamics, action,
reward, done, and reset behavior. Only the policy observation changes from the
old 21-D Isaac-specific vector to the 44-D `main_scan44` contract.
"""

import torch

from crowd_nav_rl.observations.fwmini_obs_contract import (
    CURRENT_V_IDX,
    CURRENT_W_IDX,
    LAST_CMD_0_IDX,
    LAST_CMD_1_IDX,
    MAIN_SCAN_ANGLE_MAX,
    MAIN_SCAN_ANGLE_MIN,
    MAIN_SCAN_DIM,
    MAIN_SCAN_RANGE_MAX,
    MAIN_SCAN_RANGE_MIN,
    MAIN_SCAN_SENSOR_X,
    MAIN_SCAN_SENSOR_Y,
    MAIN_SCAN_SLICE,
    OBS_DIM_MAIN_SCAN44,
    WAYPOINT_DIST_IDX,
    WAYPOINT_HEADING_IDX,
    WAYPOINT_X_IDX,
    WAYPOINT_Y_IDX,
    build_main_scan44_obs,
    normalize_ranges,
)
from crowd_nav_rl.tasks.locomotion.nav_obstacle.fwmini_nav_obstacle_real_scanlike_vec_env_cfg import (
    FWMiniNavObstacleRealScanlikeVecEnvCfg,
)
from crowd_nav_rl.tasks.locomotion.nav_obstacle.fwmini_nav_obstacle_vec_env import (
    FWMiniNavObstacleVecEnv,
)


class FWMiniNavObstacleRealScanlikeVecEnv(FWMiniNavObstacleVecEnv):
    """Obstacle env with ROS `/scan`-like policy observation.

    The policy observation is:
    - scan[0:36]: front 180-degree pseudo laser scan, normalized to [0, 1].
    - waypoint fields: currently the final goal in robot body frame.
    - current velocity fields.
    - last command fields.

    The scan is generated from `cfg.obstacles` only. It does not read Isaac
    range sensors, ROS topics, or any real robot data.
    """

    cfg: FWMiniNavObstacleRealScanlikeVecEnvCfg

    def _pseudo_scan_main_36(self) -> torch.Tensor:
        """Return raw 36-sector front scan ranges from inflated obstacle boxes."""

        self._init_buffers_and_joints()

        root_pos = self.robot.data.root_pos_w[:, :3]
        root_quat = self.robot.data.root_quat_w[:, :4]
        origins = self.scene.env_origins

        yaw = self._yaw_from_quat_wxyz(root_quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        robot_local_x = root_pos[:, 0] - origins[:, 0]
        robot_local_y = root_pos[:, 1] - origins[:, 1]

        sensor_x = robot_local_x + cos_yaw * MAIN_SCAN_SENSOR_X - sin_yaw * MAIN_SCAN_SENSOR_Y
        sensor_y = robot_local_y + sin_yaw * MAIN_SCAN_SENSOR_X + cos_yaw * MAIN_SCAN_SENSOR_Y

        angles = torch.linspace(
            MAIN_SCAN_ANGLE_MIN,
            MAIN_SCAN_ANGLE_MAX,
            MAIN_SCAN_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        ray_angles = yaw.unsqueeze(1) + angles.unsqueeze(0)
        ray_dx = torch.cos(ray_angles)
        ray_dy = torch.sin(ray_angles)

        ranges = torch.full(
            (self.num_envs, MAIN_SCAN_DIM),
            float(MAIN_SCAN_RANGE_MAX),
            dtype=torch.float32,
            device=self.device,
        )

        for cx, cy, sx, sy, _sz in self.cfg.obstacles:
            inflate = self.cfg.robot_radius + self.cfg.obstacle_collision_margin
            min_x = float(cx) - float(sx) / 2.0 - inflate
            max_x = float(cx) + float(sx) / 2.0 + inflate
            min_y = float(cy) - float(sy) / 2.0 - inflate
            max_y = float(cy) + float(sy) / 2.0 + inflate

            hit_dist = self._ray_aabb_intersection(
                sensor_x=sensor_x,
                sensor_y=sensor_y,
                ray_dx=ray_dx,
                ray_dy=ray_dy,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
            )
            ranges = torch.minimum(ranges, hit_dist)

        return torch.clamp(ranges, min=MAIN_SCAN_RANGE_MIN, max=MAIN_SCAN_RANGE_MAX)

    def _ray_aabb_intersection(
        self,
        sensor_x: torch.Tensor,
        sensor_y: torch.Tensor,
        ray_dx: torch.Tensor,
        ray_dy: torch.Tensor,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> torch.Tensor:
        """Vectorized ray vs axis-aligned box slab intersection."""

        sx = sensor_x.unsqueeze(1)
        sy = sensor_y.unsqueeze(1)

        eps = 1e-8
        inf = torch.full_like(ray_dx, float("inf"))
        neg_inf = torch.full_like(ray_dx, float("-inf"))
        parallel_x = torch.abs(ray_dx) < eps
        parallel_y = torch.abs(ray_dy) < eps
        inside_x = (sx >= min_x) & (sx <= max_x)
        inside_y = (sy >= min_y) & (sy <= max_y)
        safe_dx = torch.where(parallel_x, torch.ones_like(ray_dx), ray_dx)
        safe_dy = torch.where(parallel_y, torch.ones_like(ray_dy), ray_dy)

        tx1 = (min_x - sx) / safe_dx
        tx2 = (max_x - sx) / safe_dx
        ty1 = (min_y - sy) / safe_dy
        ty2 = (max_y - sy) / safe_dy

        tx_min = torch.minimum(tx1, tx2)
        tx_max = torch.maximum(tx1, tx2)
        ty_min = torch.minimum(ty1, ty2)
        ty_max = torch.maximum(ty1, ty2)

        tx_min = torch.where(parallel_x & inside_x, neg_inf, tx_min)
        tx_max = torch.where(parallel_x & inside_x, inf, tx_max)
        tx_min = torch.where(parallel_x & ~inside_x, inf, tx_min)
        tx_max = torch.where(parallel_x & ~inside_x, neg_inf, tx_max)
        ty_min = torch.where(parallel_y & inside_y, neg_inf, ty_min)
        ty_max = torch.where(parallel_y & inside_y, inf, ty_max)
        ty_min = torch.where(parallel_y & ~inside_y, inf, ty_min)
        ty_max = torch.where(parallel_y & ~inside_y, neg_inf, ty_max)

        t_enter = torch.maximum(tx_min, ty_min)
        t_exit = torch.minimum(tx_max, ty_max)

        hit = (
            (t_exit >= t_enter)
            & (t_exit >= MAIN_SCAN_RANGE_MIN)
            & (t_enter <= MAIN_SCAN_RANGE_MAX)
        )
        t_hit = torch.clamp(t_enter, min=MAIN_SCAN_RANGE_MIN, max=MAIN_SCAN_RANGE_MAX)
        no_hit = torch.full_like(t_hit, float(MAIN_SCAN_RANGE_MAX))
        return torch.where(hit, t_hit, no_hit)

    def _current_body_velocity_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current forward body velocity and yaw rate."""

        root_quat = self.robot.data.root_quat_w[:, :4]
        root_lin_vel = self.robot.data.root_lin_vel_w[:, :3]
        root_ang_vel = self.robot.data.root_ang_vel_w[:, :3]

        yaw = self._yaw_from_quat_wxyz(root_quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        current_v = cos_yaw * root_lin_vel[:, 0] + sin_yaw * root_lin_vel[:, 1]
        current_w = root_ang_vel[:, 2]
        return current_v, current_w

    def _get_observations(self) -> dict:
        """Build `main_scan44` policy observation.

        This observation does not expose obstacle_body_x, obstacle_body_y, or
        obstacle_clearance. Obstacle information only appears through the
        pseudo `/scan` range sectors.
        """

        self._init_buffers_and_joints()

        main_scan_raw = self._pseudo_scan_main_36()
        main_scan_norm = normalize_ranges(
            main_scan_raw,
            MAIN_SCAN_RANGE_MIN,
            MAIN_SCAN_RANGE_MAX,
        )

        waypoint_x, waypoint_y, waypoint_dist, waypoint_heading = self._goal_terms()
        current_v, current_w = self._current_body_velocity_terms()

        obs = build_main_scan44_obs(
            main_scan_norm=main_scan_norm,
            waypoint_x=waypoint_x,
            waypoint_y=waypoint_y,
            waypoint_dist=waypoint_dist,
            waypoint_heading=waypoint_heading,
            current_v=current_v,
            current_w=current_w,
            last_cmd_0=self._last_v_cmd,
            last_cmd_1=self._last_steer_cmd,
        )

        if obs.shape[-1] != OBS_DIM_MAIN_SCAN44:
            raise RuntimeError(f"main_scan44 obs dim mismatch: {obs.shape[-1]}")

        # Touch constants here so accidental slice/index drift is caught early by linters/review.
        _ = (
            MAIN_SCAN_SLICE,
            WAYPOINT_X_IDX,
            WAYPOINT_Y_IDX,
            WAYPOINT_DIST_IDX,
            WAYPOINT_HEADING_IDX,
            CURRENT_V_IDX,
            CURRENT_W_IDX,
            LAST_CMD_0_IDX,
            LAST_CMD_1_IDX,
        )

        return {"policy": obs}
