import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from crowd_nav_rl.assets.fw_mini.config.fw_mini_cfg import FW_MINI_CFG
from crowd_nav_rl.tasks.locomotion.nav_static_vec.fwmini_nav_static_vec_env_cfg import (
    FWMiniNavStaticVecEnvCfg,
)


class FWMiniNavStaticVecEnv(DirectRLEnv):
    cfg: FWMiniNavStaticVecEnvCfg

    def __init__(self, cfg: FWMiniNavStaticVecEnvCfg, render_mode: str | None = None, **kwargs):
        self.robot = None
        self.wheel_joint_ids = None
        self.steer_joint_ids = None

        self._last_v_cmd = None
        self._last_steer_cmd = None
        self._last_steer_targets = None

        self._goal_w = None
        self._prev_goal_dist = None

        self.done_reason_buf = None
        self.last_goal_dist_buf = None
        self.last_z_drop_buf = None
        self.last_max_wheel_buf = None
        self.last_max_steer_buf = None

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _setup_scene(self):
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=1.6,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            )
        )
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/DomeLight", light_cfg)

        robot_cfg = FW_MINI_CFG.replace(prim_path=self.cfg.robot_prim_path)
        self.robot = Articulation(cfg=robot_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

    def _init_buffers_and_joints(self):
        if self.wheel_joint_ids is not None:
            return

        wheel_ids, wheel_names = self.robot.find_joints(".*_wheel_joint")
        steer_ids, steer_names = self.robot.find_joints(".*_steering_hinge_joint")

        expected_wheels = [
            "left_front_wheel_joint",
            "left_rear_wheel_joint",
            "right_rear_wheel_joint",
            "right_front_wheel_joint",
        ]
        expected_steers = [
            "left_steering_hinge_joint",
            "rear_left_steering_hinge_joint",
            "rear_right_steering_hinge_joint",
            "right_steering_hinge_joint",
        ]

        if list(wheel_names) != expected_wheels:
            raise RuntimeError(f"Wheel joint order mismatch: {list(wheel_names)}")

        if list(steer_names) != expected_steers:
            raise RuntimeError(f"Steer joint order mismatch: {list(steer_names)}")

        self.wheel_joint_ids = wheel_ids
        self.steer_joint_ids = steer_ids

        self._last_v_cmd = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._last_steer_cmd = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._last_steer_targets = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        self._goal_w = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # reason code:
        # 0 none, 1 goal, 2 bad_z, 3 bad_wheel, 4 bad_steer, 5 timeout
        self.done_reason_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_goal_dist_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_z_drop_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_max_wheel_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_max_steer_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        print("wheel joints:", wheel_names)
        print("steer joints:", steer_names)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
        # q shape: [num_envs, 4], order [w, x, y, z]
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _sample_goal(self, env_ids: torch.Tensor):
        n = len(env_ids)

        gx = self.cfg.goal_x_min + (
            self.cfg.goal_x_max - self.cfg.goal_x_min
        ) * torch.rand((n,), device=self.device)

        gy = self.cfg.goal_y_min + (
            self.cfg.goal_y_max - self.cfg.goal_y_min
        ) * torch.rand((n,), device=self.device)

        origins = self.scene.env_origins[env_ids]

        self._goal_w[env_ids, 0] = origins[:, 0] + gx
        self._goal_w[env_ids, 1] = origins[:, 1] + gy

    def _goal_terms(self):
        root_pos = self.robot.data.root_pos_w[:, :3]
        root_quat = self.robot.data.root_quat_w[:, :4]

        yaw = self._yaw_from_quat_wxyz(root_quat)

        dx = self._goal_w[:, 0] - root_pos[:, 0]
        dy = self._goal_w[:, 1] - root_pos[:, 1]

        dist = torch.sqrt(dx * dx + dy * dy)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        goal_body_x = cos_yaw * dx + sin_yaw * dy
        goal_body_y = -sin_yaw * dx + cos_yaw * dy

        heading_error = torch.atan2(goal_body_y, goal_body_x)
        heading_error = self._wrap_to_pi(heading_error)

        return goal_body_x, goal_body_y, dist, heading_error

    def _compute_targets(self, v_cmd: torch.Tensor, steer_cmd: torch.Tensor, dt: float):
        cfg = self.cfg

        v_cmd = torch.clamp(v_cmd, -cfg.max_speed_mps, cfg.max_speed_mps)
        steer_cmd = torch.clamp(steer_cmd, -cfg.max_steer_rad, cfg.max_steer_rad)

        abs_steer = torch.clamp(torch.abs(steer_cmd), min=1e-4)
        sign = torch.where(steer_cmd >= 0.0, 1.0, -1.0)

        R = cfg.wheel_base / torch.tan(abs_steer)

        fl = torch.atan(cfg.wheel_base / (R - cfg.track_width / 2.0)) * sign
        fr = torch.atan(cfg.wheel_base / (R + cfg.track_width / 2.0)) * sign
        rl = -torch.atan(cfg.wheel_base / (R - cfg.track_width / 2.0)) * sign
        rr = -torch.atan(cfg.wheel_base / (R + cfg.track_width / 2.0)) * sign

        fl = torch.clamp(fl, -cfg.max_steer_rad, cfg.max_steer_rad)
        fr = torch.clamp(fr, -cfg.max_steer_rad, cfg.max_steer_rad)
        rl = torch.clamp(rl, -cfg.max_steer_rad, cfg.max_steer_rad)
        rr = torch.clamp(rr, -cfg.max_steer_rad, cfg.max_steer_rad)

        steer_targets = torch.stack([fl, rl, rr, fr], dim=1)

        straight_mask = torch.abs(steer_cmd) < 1e-6
        stop_mask = torch.abs(v_cmd) < 1e-6

        fl_v = v_cmd * (R - cfg.track_width / 2.0 * sign) / R
        fr_v = v_cmd * (R + cfg.track_width / 2.0 * sign) / R
        rl_v = v_cmd * (R - cfg.track_width / 2.0 * sign) / R
        rr_v = v_cmd * (R + cfg.track_width / 2.0 * sign) / R

        wheel_linear_turn = torch.stack([fl_v, rl_v, rr_v, fr_v], dim=1)
        wheel_linear_straight = v_cmd.unsqueeze(1).repeat(1, 4)

        wheel_linear = torch.where(
            straight_mask.unsqueeze(1),
            wheel_linear_straight,
            wheel_linear_turn,
        )

        wheel_linear = torch.where(
            stop_mask.unsqueeze(1),
            torch.zeros_like(wheel_linear),
            wheel_linear,
        )

        steer_targets = torch.where(
            straight_mask.unsqueeze(1),
            torch.zeros_like(steer_targets),
            steer_targets,
        )

        # steering rate limit
        max_delta = cfg.steer_rate_limit * dt
        delta = torch.clamp(
            steer_targets - self._last_steer_targets,
            min=-max_delta,
            max=max_delta,
        )
        self._last_steer_targets = self._last_steer_targets + delta

        wheel_omegas = wheel_linear / cfg.wheel_radius

        return self._last_steer_targets, wheel_omegas

    def _pre_physics_step(self, actions: torch.Tensor):
        self._init_buffers_and_joints()

        actions = torch.clamp(actions, -1.0, 1.0)

        v_action = torch.clamp(actions[:, 0], 0.0, 1.0)
        steer_action = actions[:, 1]

        v_cmd = v_action * self.cfg.action_v_scale
        steer_cmd = steer_action * self.cfg.action_steer_scale

        warmup_mask = self.episode_length_buf < self.cfg.warmup_steps

        v_cmd = torch.where(warmup_mask, torch.zeros_like(v_cmd), v_cmd)
        steer_cmd = torch.where(warmup_mask, torch.zeros_like(steer_cmd), steer_cmd)

        self._last_v_cmd = v_cmd
        self._last_steer_cmd = steer_cmd

    def _apply_action(self):
        self._init_buffers_and_joints()

        dt = self.physics_dt * self.cfg.decimation

        steer_targets, wheel_omegas = self._compute_targets(
            self._last_v_cmd,
            self._last_steer_cmd,
            dt,
        )

        self.robot.set_joint_position_target(steer_targets, joint_ids=self.steer_joint_ids)
        self.robot.set_joint_velocity_target(wheel_omegas, joint_ids=self.wheel_joint_ids)
        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        self._init_buffers_and_joints()

        root_pos = self.robot.data.root_pos_w[:, :3]
        root_lin_vel = self.robot.data.root_lin_vel_w[:, :3]
        root_ang_vel = self.robot.data.root_ang_vel_w[:, :3]

        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids]
        steer_pos = self.robot.data.joint_pos[:, self.steer_joint_ids]

        goal_body_x, goal_body_y, goal_dist, heading_error = self._goal_terms()

        obs = torch.cat(
            [
                self._last_v_cmd.unsqueeze(1),
                self._last_steer_cmd.unsqueeze(1),
                goal_body_x.unsqueeze(1),
                goal_body_y.unsqueeze(1),
                goal_dist.unsqueeze(1),
                heading_error.unsqueeze(1),
                root_pos[:, 2:3],
                root_lin_vel[:, 0:1],
                root_lin_vel[:, 1:2],
                root_ang_vel[:, 2:3],
                wheel_vel,
                steer_pos,
            ],
            dim=1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._init_buffers_and_joints()

        _, _, goal_dist, heading_error = self._goal_terms()

        progress = self._prev_goal_dist - goal_dist
        self._prev_goal_dist = goal_dist.detach()

        action_penalty = self._last_v_cmd * self._last_v_cmd + self._last_steer_cmd * self._last_steer_cmd

        reward = (
            self.cfg.progress_reward_scale * progress
            - self.cfg.distance_penalty_scale * goal_dist
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.steer_penalty_scale * torch.abs(heading_error)
        )

        reached_goal = goal_dist < self.cfg.goal_radius
        reward = torch.where(reached_goal, reward + self.cfg.goal_reward, reward)

        warmup_mask = self.episode_length_buf < self.cfg.warmup_steps
        reward = torch.where(warmup_mask, torch.zeros_like(reward), reward)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._init_buffers_and_joints()

        root_z = self.robot.data.root_pos_w[:, 2]
        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids]
        steer_pos = self.robot.data.joint_pos[:, self.steer_joint_ids]

        _, _, goal_dist, _ = self._goal_terms()

        z_drop = torch.abs(root_z - self.cfg.settled_z_ref)
        max_wheel = torch.max(torch.abs(wheel_vel), dim=1).values
        max_steer = torch.max(torch.abs(steer_pos), dim=1).values

        reached_goal = goal_dist < self.cfg.goal_radius
        bad_z = z_drop > self.cfg.collapse_z_drop
        bad_wheel = max_wheel > self.cfg.max_wheel_vel
        bad_steer = max_steer > self.cfg.max_steer_pos

        warmup_mask = self.episode_length_buf < self.cfg.warmup_steps

        terminated = reached_goal | bad_z | bad_wheel | bad_steer
        terminated = torch.where(warmup_mask, torch.zeros_like(terminated), terminated)

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        self.done_reason_buf[:] = 0
        self.done_reason_buf = torch.where(reached_goal, torch.ones_like(self.done_reason_buf) * 1, self.done_reason_buf)
        self.done_reason_buf = torch.where(bad_z, torch.ones_like(self.done_reason_buf) * 2, self.done_reason_buf)
        self.done_reason_buf = torch.where(bad_wheel, torch.ones_like(self.done_reason_buf) * 3, self.done_reason_buf)
        self.done_reason_buf = torch.where(bad_steer, torch.ones_like(self.done_reason_buf) * 4, self.done_reason_buf)
        self.done_reason_buf = torch.where(time_out, torch.ones_like(self.done_reason_buf) * 5, self.done_reason_buf)

        self.last_goal_dist_buf = goal_dist.detach()
        self.last_z_drop_buf = z_drop.detach()
        self.last_max_wheel_buf = max_wheel.detach()
        self.last_max_steer_buf = max_steer.detach()

        return terminated, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        self._init_buffers_and_joints()

        origins = self.scene.env_origins[env_ids]

        root_state = self.robot.data.default_root_state[env_ids].clone()

        root_state[:, 0] = origins[:, 0]
        root_state[:, 1] = origins[:, 1]
        root_state[:, 2] = origins[:, 2] + 0.20

        root_state[:, 3] = 1.0
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = 0.0

        root_state[:, 7:13] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:] = 0.0
        joint_vel[:] = 0.0

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._last_v_cmd[env_ids] = 0.0
        self._last_steer_cmd[env_ids] = 0.0
        self._last_steer_targets[env_ids, :] = 0.0

        self._sample_goal(env_ids)

        _, _, goal_dist, _ = self._goal_terms()
        self._prev_goal_dist[env_ids] = goal_dist[env_ids].detach()