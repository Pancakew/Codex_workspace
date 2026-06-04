import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from crowd_nav_rl.assets.fw_mini.config.fw_mini_cfg import FW_MINI_CFG
from crowd_nav_rl.controllers.fwmini_cmdvel_controller import (
    FWMiniCmdVelController,
    FWMiniCmdVelConfig,
)
from crowd_nav_rl.tasks.locomotion.nav_static.fwmini_nav_static_env_cfg import (
    FWMiniNavStaticEnvCfg,
)


class FWMiniNavStaticEnv(DirectRLEnv):
    cfg: FWMiniNavStaticEnvCfg

    def __init__(self, cfg: FWMiniNavStaticEnvCfg, render_mode: str | None = None, **kwargs):
        self.robot = None
        self.wheel_joint_ids = None
        self.steer_joint_ids = None
        self.cmd_controller = None

        self._last_v_cmd = 0.0
        self._last_steer_cmd = 0.0
        self._settled_z = cfg.settled_z_ref

        self._goal_w = None
        self._prev_goal_dist = None

        # 关键：done reason 只由 _get_dones() 写入，reset 不再覆盖
        self._last_done_reason = "none"
        self._last_goal_dist = 999.0
        self._last_max_wheel = 0.0
        self._last_max_steer = 0.0
        self._last_z_drop = 0.0

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

    def _init_joints_and_controller(self):
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

        self.cmd_controller = FWMiniCmdVelController(
            FWMiniCmdVelConfig(
                max_speed_mps=self.cfg.max_speed_mps,
                max_steer_rad=self.cfg.max_steer_rad,
                wheel_radius=self.cfg.wheel_radius,
                wheel_base=self.cfg.wheel_base,
                track_width=self.cfg.track_width,
                steer_rate_limit=self.cfg.steer_rate_limit,
            )
        )
        self.cmd_controller.reset()

        print("wheel joints:", wheel_names)
        print("steer joints:", steer_names)

    @staticmethod
    def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[0], q[1], q[2], q[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _sample_goal(self):
        gx = self.cfg.goal_x_min + (
            self.cfg.goal_x_max - self.cfg.goal_x_min
        ) * torch.rand((1,), device=self.device)

        gy = self.cfg.goal_y_min + (
            self.cfg.goal_y_max - self.cfg.goal_y_min
        ) * torch.rand((1,), device=self.device)

        self._goal_w = torch.stack([gx, gy], dim=-1)

    def _goal_terms(self):
        root_pos = self.robot.data.root_pos_w[0, :3]
        root_quat = self.robot.data.root_quat_w[0, :4]

        yaw = self._yaw_from_quat_wxyz(root_quat)

        dx = self._goal_w[0, 0] - root_pos[0]
        dy = self._goal_w[0, 1] - root_pos[1]

        dist = torch.sqrt(dx * dx + dy * dy)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        goal_body_x = cos_yaw * dx + sin_yaw * dy
        goal_body_y = -sin_yaw * dx + cos_yaw * dy

        heading_error = torch.atan2(goal_body_y, goal_body_x)
        heading_error = self._wrap_to_pi(heading_error)

        return goal_body_x, goal_body_y, dist, heading_error

    def _pre_physics_step(self, actions: torch.Tensor):
        self._init_joints_and_controller()

        if self.episode_length_buf[0].item() < self.cfg.warmup_steps:
            self._last_v_cmd = 0.0
            self._last_steer_cmd = 0.0
            return

        a_v = torch.clamp(actions[0, 0], -1.0, 1.0)
        a_s = torch.clamp(actions[0, 1], -1.0, 1.0)

        # 当前阶段只允许前进，不允许倒车
        a_v = torch.clamp(a_v, 0.0, 1.0)

        self._last_v_cmd = float((a_v * self.cfg.action_v_scale).item())
        self._last_steer_cmd = float((a_s * self.cfg.action_steer_scale).item())

    def _apply_action(self):
        self._init_joints_and_controller()

        dt = self.physics_dt * self.cfg.decimation

        steer_angles, wheel_omegas = self.cmd_controller.step(
            v_cmd=self._last_v_cmd,
            steer_cmd=self._last_steer_cmd,
            dt=dt,
            gear=6,
        )

        steer_tensor = torch.tensor([steer_angles], dtype=torch.float32, device=self.device)
        wheel_tensor = torch.tensor([wheel_omegas], dtype=torch.float32, device=self.device)

        self.robot.set_joint_position_target(steer_tensor, joint_ids=self.steer_joint_ids)
        self.robot.set_joint_velocity_target(wheel_tensor, joint_ids=self.wheel_joint_ids)
        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        self._init_joints_and_controller()

        root_pos = self.robot.data.root_pos_w[0, :3]
        root_lin_vel = self.robot.data.root_lin_vel_w[0, :3]
        root_ang_vel = self.robot.data.root_ang_vel_w[0, :3]

        wheel_vel = self.robot.data.joint_vel[0, self.wheel_joint_ids].reshape(-1)
        steer_pos = self.robot.data.joint_pos[0, self.steer_joint_ids].reshape(-1)

        goal_body_x, goal_body_y, goal_dist, heading_error = self._goal_terms()

        obs = torch.cat(
            [
                torch.tensor(
                    [
                        self._last_v_cmd,
                        self._last_steer_cmd,
                        goal_body_x.item(),
                        goal_body_y.item(),
                        goal_dist.item(),
                        heading_error.item(),
                        root_pos[2].item(),
                        root_lin_vel[0].item(),
                        root_lin_vel[1].item(),
                        root_ang_vel[2].item(),
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
                wheel_vel.to(self.device),
                steer_pos.to(self.device),
            ],
            dim=0,
        ).unsqueeze(0)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        if self.episode_length_buf[0].item() < self.cfg.warmup_steps:
            return torch.zeros(1, dtype=torch.float32, device=self.device)

        _, _, goal_dist, heading_error = self._goal_terms()

        if self._prev_goal_dist is None:
            self._prev_goal_dist = goal_dist.detach()

        progress = self._prev_goal_dist - goal_dist
        self._prev_goal_dist = goal_dist.detach()

        action_penalty = (
            self._last_v_cmd * self._last_v_cmd
            + self._last_steer_cmd * self._last_steer_cmd
        )

        reward = (
            self.cfg.progress_reward_scale * progress
            - self.cfg.distance_penalty_scale * goal_dist
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.steer_penalty_scale * torch.abs(heading_error)
        )

        if goal_dist < self.cfg.goal_radius:
            reward = reward + self.cfg.goal_reward

        return reward.reshape(1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._init_joints_and_controller()

        if self.episode_length_buf[0].item() < self.cfg.warmup_steps:
            self._last_done_reason = "none"
            terminated = torch.zeros(1, dtype=torch.bool, device=self.device)
            time_out = self.episode_length_buf >= (self.max_episode_length - 1)
            return terminated, time_out

        root_z = self.robot.data.root_pos_w[0, 2]
        wheel_vel = self.robot.data.joint_vel[0, self.wheel_joint_ids].reshape(-1)
        steer_pos = self.robot.data.joint_pos[0, self.steer_joint_ids].reshape(-1)

        _, _, goal_dist, _ = self._goal_terms()

        z_drop = abs(float(self._settled_z) - float(root_z.item()))
        max_wheel = float(torch.max(torch.abs(wheel_vel)).item())
        max_steer = float(torch.max(torch.abs(steer_pos)).item())

        self._last_goal_dist = float(goal_dist.item())
        self._last_max_wheel = max_wheel
        self._last_max_steer = max_steer
        self._last_z_drop = z_drop

        reached_goal = bool(goal_dist.item() < self.cfg.goal_radius)
        bad_z = z_drop > self.cfg.collapse_z_drop
        bad_wheel = max_wheel > self.cfg.max_wheel_vel
        bad_steer = max_steer > self.cfg.max_steer_pos

        if reached_goal:
            self._last_done_reason = "goal"
        elif bad_z:
            self._last_done_reason = "bad_z"
        elif bad_wheel:
            self._last_done_reason = "bad_wheel"
        elif bad_steer:
            self._last_done_reason = "bad_steer"
        else:
            self._last_done_reason = "none"

        terminated = torch.tensor(
            [reached_goal or bad_z or bad_wheel or bad_steer],
            dtype=torch.bool,
            device=self.device,
        )

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        if bool(time_out.item()):
            self._last_done_reason = "timeout"

        return terminated, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        self._init_joints_and_controller()
        self.cmd_controller.reset()

        root_state = self.robot.data.default_root_state[env_ids].clone()

        root_state[:, 0] = 0.0
        root_state[:, 1] = 0.0
        root_state[:, 2] = 0.20

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

        zero_wheel = torch.zeros(
            (len(env_ids), len(self.wheel_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        zero_steer = torch.zeros(
            (len(env_ids), len(self.steer_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )

        self.robot.set_joint_velocity_target(
            zero_wheel,
            joint_ids=self.wheel_joint_ids,
            env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            zero_steer,
            joint_ids=self.steer_joint_ids,
            env_ids=env_ids,
        )
        self.robot.write_data_to_sim()

        self._sample_goal()

        self._prev_goal_dist = None
        self._last_v_cmd = 0.0
        self._last_steer_cmd = 0.0
        self._settled_z = self.cfg.settled_z_ref

        # 注意：这里不再写 self._last_done_reason = "reset"
        # 否则会覆盖刚刚结束 episode 的真正原因。