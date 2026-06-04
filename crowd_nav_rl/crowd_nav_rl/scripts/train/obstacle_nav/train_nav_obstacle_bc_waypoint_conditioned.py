"""Waypoint-conditioned behavior cloning for FW-mini obstacle navigation.

Learning target:
- A hand-written waypoint expert drives around the obstacle.
- The script records `(observation, expert_action)` pairs.
- A neural network learns to imitate the expert action.

Policy input:
- 21-D environment observation from `FWMiniNavObstacleVecEnv`.
- 6-D waypoint feature: target body x/y, target distance/heading, phase, pass side.

Policy output:
- 2-D action: `[forward_speed_action, steer_action]`.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Waypoint-conditioned BC for FW-mini obstacle navigation.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--collect-steps", type=int, default=6000)
parser.add_argument("--train-epochs", type=int, default=120)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--save-dir", type=str, default="logs/fwmini_obstacle_bc_wp")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--print-every", type=int, default=10)

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
    """Make data collection and network initialization reproducible."""

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
    """Convert a world-frame waypoint into the robot body frame."""

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
    """Per-env waypoint memory used by the hand-written expert.

    `phase` selects which waypoint is active:
    - 0: move to a side lane near the obstacle.
    - 1: pass the obstacle along that lane.
    - 2: drive to the final sampled goal.

    `pass_side` is +1 for upper lane and -1 for lower lane.
    """

    def __init__(self, num_envs, device):
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pass_side = torch.ones(num_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids):
        self.phase[env_ids] = 0
        self.pass_side[env_ids] = 1.0


def waypoint_expert(env, obs_tensor, state):
    """Return expert action and 6-D waypoint features for every env.

    This is not learned. It is the teacher used to generate the BC dataset.
    """

    device = obs_tensor.device
    num_envs = obs_tensor.shape[0]

    origins = env.scene.env_origins
    root_pos = env.robot.data.root_pos_w[:, :3]

    goal_dist = obs_tensor[:, 4]

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

    target_body_x, target_body_y, target_dist, target_heading = body_target_from_world(env, target)

    abs_heading = torch.abs(target_heading)

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
    steer_action = torch.clamp(target_heading / 0.25, -1.0, 1.0)

    action = torch.stack([v_action, steer_action], dim=1)

    wp_features = torch.stack(
        [
            target_body_x,
            target_body_y,
            target_dist,
            target_heading,
            state.phase.float() / 2.0,
            state.pass_side,
        ],
        dim=1,
    )

    return action, wp_features


class BCPolicy(nn.Module):
    """Small MLP policy trained with mean-squared error to expert actions."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()

        layers = []
        last_dim = obs_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h

        layers.append(nn.Linear(last_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_norm):
        return torch.tanh(self.net(obs_norm))


def main():
    """Collect expert data, normalize observations, train BC, save checkpoint."""

    set_seed(args_cli.seed)
    os.makedirs(args_cli.save_dir, exist_ok=True)

    cfg = FWMiniNavObstacleVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.use_fixed_goal = False

    env = FWMiniNavObstacleVecEnv(cfg=cfg)
    obs, _ = env.reset()
    obs_tensor = extract_policy_obs(obs)

    device = env.device

    state = WaypointState(num_envs=args_cli.num_envs, device=device)

    obs_chunks = []
    act_chunks = []

    total_episodes = 0
    success = 0
    collision = 0
    timeout = 0
    failure = 0

    print("[INFO] Start collecting waypoint-conditioned obstacle expert data")
    print(f"[INFO] base_obs_dim={obs_tensor.shape[1]}, extra_wp_dim=6, final_obs_dim={obs_tensor.shape[1] + 6}")
    print(f"[INFO] num_envs={args_cli.num_envs}")
    print("[INFO] obstacles:", cfg.obstacles)

    for step in range(args_cli.collect_steps):
        # Data collection phase: run the expert in the real environment.
        with torch.no_grad():
            expert_action, wp_features = waypoint_expert(env, obs_tensor, state)
            obs_aug = torch.cat([obs_tensor, wp_features], dim=1)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        expert_action[warmup_mask, :] = 0.0

        valid_mask = ~warmup_mask

        if torch.any(valid_mask):
            obs_chunks.append(obs_aug[valid_mask].detach().clone())
            act_chunks.append(expert_action[valid_mask].detach().clone())

        obs, rew, terminated, truncated, info = env.step(expert_action)
        obs_tensor = extract_policy_obs(obs)

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if len(done_ids) > 0:
            reasons = env.done_reason_buf[done_ids].detach().cpu().tolist()

            for reason_code in reasons:
                total_episodes += 1
                reason = REASON_MAP.get(int(reason_code), "unknown")

                if reason == "goal":
                    success += 1
                elif reason == "collision":
                    collision += 1
                elif reason == "timeout":
                    timeout += 1
                else:
                    failure += 1

            state.reset(done_ids)

        if step % 500 == 0:
            print(
                f"[COLLECT] step={step} | "
                f"samples={sum(x.shape[0] for x in obs_chunks)} | "
                f"episodes={total_episodes} | "
                f"success={success} | "
                f"collision={collision} | "
                f"timeout={timeout} | "
                f"failure={failure} | "
                f"success_rate={success / max(total_episodes, 1):.3f} | "
                f"mean_dist={obs_tensor[:, 4].mean().item():.4f}"
            )

    dataset_obs = torch.cat(obs_chunks, dim=0)
    dataset_act = torch.cat(act_chunks, dim=0)

    obs_mean = dataset_obs.mean(dim=0, keepdim=True)
    obs_std = dataset_obs.std(dim=0, keepdim=True).clamp_min(1e-6)

    dataset_obs_norm = (dataset_obs - obs_mean) / obs_std

    obs_dim = dataset_obs.shape[1]
    act_dim = cfg.action_space

    model = BCPolicy(obs_dim=obs_dim, act_dim=act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args_cli.lr)

    dataset_size = dataset_obs_norm.shape[0]

    print("[INFO] Expert data collected")
    print(f"[INFO] dataset_obs={dataset_obs.shape}, dataset_act={dataset_act.shape}")
    print("[INFO] Start waypoint-conditioned BC training")

    for epoch in range(args_cli.train_epochs):
        # Supervised learning phase: fit policy(obs_aug) ~= expert_action.
        perm = torch.randperm(dataset_size, device=device)

        total_loss = 0.0
        total_batches = 0

        for start in range(0, dataset_size, args_cli.batch_size):
            idx = perm[start : start + args_cli.batch_size]

            batch_obs = dataset_obs_norm[idx]
            batch_act = dataset_act[idx]

            pred = model(batch_obs)
            loss = torch.mean((pred - batch_act) ** 2)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        if epoch % args_cli.print_every == 0 or epoch == args_cli.train_epochs - 1:
            with torch.no_grad():
                n_eval = min(16384, dataset_size)
                pred = model(dataset_obs_norm[:n_eval])
                target = dataset_act[:n_eval]
                eval_mse = torch.mean((pred - target) ** 2).item()
                eval_mae = torch.mean(torch.abs(pred - target)).item()

            print(
                f"[BC EPOCH {epoch + 1:04d}] "
                f"train_loss={total_loss / max(total_batches, 1):.6f} | "
                f"eval_mse={eval_mse:.6f} | "
                f"eval_mae={eval_mae:.6f}"
            )

    ckpt = {
        "model": model.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dims": [256, 256],
        "base_obs_dim": obs_tensor.shape[1],
        "wp_dim": 6,
        "task": "fwmini_obstacle_bc_waypoint_conditioned",
        "seed": args_cli.seed,
    }

    latest_path = os.path.join(args_cli.save_dir, "latest.pt")
    policy_path = os.path.join(args_cli.save_dir, "obstacle_bc_wp_policy.pt")

    torch.save(ckpt, latest_path)
    torch.save(ckpt, policy_path)

    print(f"[INFO] Saved checkpoint: {latest_path}")
    print(f"[INFO] Saved checkpoint: {policy_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
