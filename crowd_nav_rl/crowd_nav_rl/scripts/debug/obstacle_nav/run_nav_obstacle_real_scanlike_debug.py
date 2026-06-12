import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug FW-mini obstacle env with main_scan44 observation.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from crowd_nav_rl.observations.fwmini_obs_contract import (
    CURRENT_V_IDX,
    CURRENT_W_IDX,
    LAST_CMD_0_IDX,
    LAST_CMD_1_IDX,
    MAIN_SCAN_DIM,
    MAIN_SCAN_SLICE,
    OBS_DIM_MAIN_SCAN44,
    PROFILE_MAIN_SCAN44,
    WAYPOINT_DIST_IDX,
    WAYPOINT_HEADING_IDX,
)
from crowd_nav_rl.tasks.locomotion.nav_obstacle.fwmini_nav_obstacle_real_scanlike_vec_env_cfg import (
    FWMiniNavObstacleRealScanlikeVecEnvCfg,
)
from crowd_nav_rl.tasks.locomotion.nav_obstacle.fwmini_nav_obstacle_real_scanlike_vec_env import (
    FWMiniNavObstacleRealScanlikeVecEnv,
)


REASON_MAP = {
    0: "none",
    1: "goal",
    2: "collision",
    3: "bad_z",
    4: "bad_wheel",
    5: "bad_steer",
    6: "timeout",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_from_quat_wxyz(q):
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def body_target_from_world(env, target_w):
    root_pos = env.robot.data.root_pos_w[:, :3]
    root_quat = env.robot.data.root_quat_w[:, :4]

    yaw = yaw_from_quat_wxyz(root_quat)

    dx = target_w[:, 0] - root_pos[:, 0]
    dy = target_w[:, 1] - root_pos[:, 1]

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    body_x = cos_yaw * dx + sin_yaw * dy
    body_y = -sin_yaw * dx + cos_yaw * dy
    heading = wrap_to_pi(torch.atan2(body_y, body_x))

    return heading


class VecWaypointState:
    def __init__(self, num_envs, device):
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids):
        self.phase[env_ids] = 0


def fixed_upper_waypoint_policy(env, obs_tensor, state):
    device = obs_tensor.device
    num_envs = obs_tensor.shape[0]

    origins = env.scene.env_origins
    root_pos = env.robot.data.root_pos_w[:, :3]

    goal_dist = obs_tensor[:, WAYPOINT_DIST_IDX]

    local_x = root_pos[:, 0] - origins[:, 0]
    local_y = root_pos[:, 1] - origins[:, 1]

    obs_cx, _obs_cy, _obs_sx, _obs_sy, _obs_sz = env.cfg.obstacles[0]

    p0 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    p1 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)

    p0[:, 0] = origins[:, 0] + obs_cx + 0.15
    p0[:, 1] = origins[:, 1] + 1.05

    p1[:, 0] = origins[:, 0] + obs_cx + 1.35
    p1[:, 1] = origins[:, 1] + 1.05

    p2 = env._goal_w.clone()

    to_phase1 = (state.phase == 0) & (local_x > obs_cx - 0.05) & (local_y > 0.75)
    state.phase = torch.where(to_phase1, torch.ones_like(state.phase), state.phase)

    to_phase2 = (state.phase == 1) & (local_x > obs_cx + 1.10)
    state.phase = torch.where(to_phase2, torch.ones_like(state.phase) * 2, state.phase)

    target = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    target = torch.where((state.phase == 0).unsqueeze(1), p0, target)
    target = torch.where((state.phase == 1).unsqueeze(1), p1, target)
    target = torch.where((state.phase == 2).unsqueeze(1), p2, target)

    heading = body_target_from_world(env, target)
    abs_heading = torch.abs(heading)

    v_action = torch.where(
        abs_heading > 1.0,
        torch.full_like(abs_heading, 0.10),
        torch.where(
            abs_heading > 0.55,
            torch.full_like(abs_heading, 0.22),
            torch.full_like(abs_heading, 0.45),
        ),
    )
    v_action = torch.where(goal_dist < env.cfg.goal_radius, torch.zeros_like(v_action), v_action)

    steer_action = torch.clamp(heading / env.cfg.action_steer_scale, -1.0, 1.0)
    return torch.stack([v_action, steer_action], dim=1), target


def extract_policy_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def validate_obs_shape(obs_tensor):
    if obs_tensor.shape[-1] != OBS_DIM_MAIN_SCAN44:
        raise RuntimeError(
            f"Expected obs['policy'].shape[-1] == {OBS_DIM_MAIN_SCAN44}, got {obs_tensor.shape[-1]}"
        )


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavObstacleRealScanlikeVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.use_fixed_goal = True
    cfg.fixed_goal_x = 4.8
    cfg.fixed_goal_y = 0.8

    print("[INFO] profile:", PROFILE_MAIN_SCAN44)
    print("[INFO] obs_dim:", OBS_DIM_MAIN_SCAN44)
    print("[INFO] current LAST_CMD_1 meaning: steer_angle_cmd")
    print("[INFO] future ROS [v,w] profile LAST_CMD_1 meaning: yaw_rate_cmd")
    print("[INFO] pseudo scan: front 180 deg, main /scan only, scan dim:", MAIN_SCAN_DIM)
    print("[INFO] obstacles:", cfg.obstacles)

    env = FWMiniNavObstacleRealScanlikeVecEnv(cfg=cfg)
    obs, _ = env.reset()
    obs_tensor = extract_policy_obs(obs)
    validate_obs_shape(obs_tensor)

    state = VecWaypointState(num_envs=args_cli.num_envs, device=env.device)
    reason_counts = {name: 0 for name in REASON_MAP.values()}

    front_idx = MAIN_SCAN_DIM // 2
    right_front_idx = MAIN_SCAN_DIM // 4
    left_front_idx = (3 * MAIN_SCAN_DIM) // 4

    for step in range(args_cli.steps):
        action, _target = fixed_upper_waypoint_policy(env, obs_tensor, state)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        action[warmup_mask, :] = 0.0

        obs, _rew, terminated, truncated, _info = env.step(action)
        obs_tensor = extract_policy_obs(obs)
        validate_obs_shape(obs_tensor)

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)
        if len(done_ids) > 0:
            reasons = env.done_reason_buf[done_ids].detach().cpu().tolist()
            for reason_code in reasons:
                reason = REASON_MAP.get(int(reason_code), "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            state.reset(done_ids)

        if step % args_cli.print_every == 0:
            o0 = obs_tensor[0]
            scan0 = o0[MAIN_SCAN_SLICE]
            print(
                f"step={step:04d} | "
                f"obs_shape={tuple(obs_tensor.shape)} | "
                f"min_scan={scan0.min().item():.3f} | "
                f"front_scan={scan0[front_idx].item():.3f} | "
                f"left_front_scan={scan0[left_front_idx].item():.3f} | "
                f"right_front_scan={scan0[right_front_idx].item():.3f} | "
                f"waypoint_dist={o0[WAYPOINT_DIST_IDX].item():.3f} | "
                f"waypoint_heading={o0[WAYPOINT_HEADING_IDX].item():.3f} | "
                f"current_v={o0[CURRENT_V_IDX].item():.3f} | "
                f"current_w={o0[CURRENT_W_IDX].item():.3f} | "
                f"last_cmd_0={o0[LAST_CMD_0_IDX].item():.3f} | "
                f"last_cmd_1={o0[LAST_CMD_1_IDX].item():.3f} | "
                f"done_counts={reason_counts}"
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
