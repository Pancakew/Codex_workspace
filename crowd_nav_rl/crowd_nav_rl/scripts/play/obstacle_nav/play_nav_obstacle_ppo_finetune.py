import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play PPO fine-tuned waypoint-conditioned obstacle policy.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--checkpoint", type=str, default="logs/fwmini_obstacle_ppo_bc_wp/latest.pt")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--print-every", type=int, default=100)
parser.add_argument("--seed", type=int, default=123)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from crowd_nav_rl.tasks.locomotion.nav_obstacle.fwmini_nav_obstacle_vec_env_cfg import (
    FWMiniNavObstacleVecEnvCfg,
)
from crowd_nav_rl.tasks.locomotion.nav_obstacle.fwmini_nav_obstacle_vec_env import (
    FWMiniNavObstacleVecEnv,
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
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
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

    dist = torch.sqrt(body_x * body_x + body_y * body_y)
    heading = wrap_to_pi(torch.atan2(body_y, body_x))

    return body_x, body_y, dist, heading


def extract_policy_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


class WaypointState:
    def __init__(self, num_envs, device):
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pass_side = torch.ones(num_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids):
        self.phase[env_ids] = 0
        self.pass_side[env_ids] = 1.0


def waypoint_features(env, obs_tensor, state):
    device = obs_tensor.device
    num_envs = obs_tensor.shape[0]

    origins = env.scene.env_origins
    root_pos = env.robot.data.root_pos_w[:, :3]

    local_x = root_pos[:, 0] - origins[:, 0]
    local_y = root_pos[:, 1] - origins[:, 1]

    obs_cx, obs_cy, obs_sx, obs_sy, _ = env.cfg.obstacles[0]

    goal_local_y = env._goal_w[:, 1] - origins[:, 1]
    new_side = torch.where(goal_local_y >= 0.0, 1.0, -1.0)

    state.pass_side = torch.where(state.phase == 0, new_side, state.pass_side)

    lane_y = state.pass_side * 1.05

    p0 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    p1 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)

    p0[:, 0] = origins[:, 0] + obs_cx + 0.15
    p0[:, 1] = origins[:, 1] + lane_y

    p1[:, 0] = origins[:, 0] + obs_cx + 1.35
    p1[:, 1] = origins[:, 1] + lane_y

    p2 = env._goal_w.clone()

    reached_lane = torch.abs(local_y - lane_y) < 0.25
    reached_x_near_obs = local_x > (obs_cx - 0.05)

    to_phase1 = (state.phase == 0) & reached_lane & reached_x_near_obs
    state.phase = torch.where(to_phase1, torch.ones_like(state.phase), state.phase)

    to_phase2 = (state.phase == 1) & (local_x > obs_cx + 1.10)
    state.phase = torch.where(to_phase2, torch.ones_like(state.phase) * 2, state.phase)

    target = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    target = torch.where((state.phase == 0).unsqueeze(1), p0, target)
    target = torch.where((state.phase == 1).unsqueeze(1), p1, target)
    target = torch.where((state.phase == 2).unsqueeze(1), p2, target)

    tx, ty, td, th = body_target_from_world(env, target)

    return torch.stack(
        [
            tx,
            ty,
            td,
            th,
            state.phase.float() / 2.0,
            state.pass_side,
        ],
        dim=1,
    )


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()

        layers = []
        last = obs_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h

        layers.append(nn.Linear(last, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_norm):
        return torch.tanh(self.net(obs_norm))


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavObstacleVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.use_fixed_goal = False

    env = FWMiniNavObstacleVecEnv(cfg=cfg)
    obs, _ = env.reset()
    obs_tensor = extract_policy_obs(obs)

    device = env.device

    ckpt = torch.load(args_cli.checkpoint, map_location=device)

    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    hidden_dims = tuple(ckpt.get("hidden_dims", [256, 256]))

    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)

    state = WaypointState(num_envs=args_cli.num_envs, device=device)

    total_episodes = 0
    success = 0
    collision = 0
    timeout = 0
    bad_z = 0
    bad_wheel = 0
    bad_steer = 0
    unknown_failure = 0

    print("[INFO] Start obstacle PPO fine-tuned policy play")
    print(f"[INFO] checkpoint={args_cli.checkpoint}")
    print(f"[INFO] num_envs={args_cli.num_envs}")
    print("[INFO] obstacles:", cfg.obstacles)

    for step in range(args_cli.steps):
        with torch.no_grad():
            wp = waypoint_features(env, obs_tensor, state)
            obs_aug = torch.cat([obs_tensor, wp], dim=1)
            obs_norm = (obs_aug - obs_mean) / obs_std

            action = actor(obs_norm)
            action = torch.clamp(action, -1.0, 1.0)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        action[warmup_mask, :] = 0.0

        obs, rew, terminated, truncated, info = env.step(action)
        obs_tensor = extract_policy_obs(obs)

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if step % args_cli.print_every == 0:
            o0 = obs_tensor[0]
            print(
                f"step={step:04d} | "
                f"phase_count=[{int((state.phase == 0).sum().item())}, "
                f"{int((state.phase == 1).sum().item())}, "
                f"{int((state.phase == 2).sum().item())}] | "
                f"env0_dist={o0[4].item():.3f} | "
                f"env0_obs=({o0[18].item():.3f},{o0[19].item():.3f},{o0[20].item():.3f}) | "
                f"mean_action=({action[:, 0].mean().item():.3f},{action[:, 1].mean().item():.3f}) | "
                f"episodes={total_episodes} | "
                f"success={success} | "
                f"collision={collision} | "
                f"timeout={timeout} | "
                f"bad_z={bad_z} | "
                f"bad_wheel={bad_wheel} | "
                f"bad_steer={bad_steer}"
            )

        if len(done_ids) > 0:
            reasons = env.done_reason_buf[done_ids].detach().cpu().tolist()

            for reason_code in reasons:
                reason = REASON_MAP.get(int(reason_code), "unknown")
                total_episodes += 1

                if reason == "goal":
                    success += 1
                elif reason == "collision":
                    collision += 1
                elif reason == "timeout":
                    timeout += 1
                elif reason == "bad_z":
                    bad_z += 1
                elif reason == "bad_wheel":
                    bad_wheel += 1
                elif reason == "bad_steer":
                    bad_steer += 1
                else:
                    unknown_failure += 1

            state.reset(done_ids)

    failure = bad_z + bad_wheel + bad_steer + unknown_failure

    print("\n================ OBSTACLE PPO FINETUNE PLAY SUMMARY ================")
    print(f"checkpoint     : {args_cli.checkpoint}")
    print(f"num_envs       : {args_cli.num_envs}")
    print(f"steps          : {args_cli.steps}")
    print(f"episodes       : {total_episodes}")
    print(f"success        : {success}")
    print(f"collision      : {collision}")
    print(f"timeout        : {timeout}")
    print(f"bad_z          : {bad_z}")
    print(f"bad_wheel      : {bad_wheel}")
    print(f"bad_steer      : {bad_steer}")
    print(f"unknown_failure: {unknown_failure}")
    print(f"failure        : {failure}")

    if total_episodes > 0:
        print(f"success_rate   : {success / total_episodes:.3f}")
        print(f"collision_rate : {collision / total_episodes:.3f}")
        print(f"timeout_rate   : {timeout / total_episodes:.3f}")

    print("====================================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()