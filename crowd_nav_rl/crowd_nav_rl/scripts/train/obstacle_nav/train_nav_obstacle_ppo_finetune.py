"""PPO fine-tuning for waypoint-conditioned FW-mini obstacle navigation.

This script starts from a trained waypoint-conditioned BC policy, then improves
it with environment reward using PPO.

Learning target:
- Actor learns actions that maximize navigation reward.
- Critic learns value estimates for PPO advantage calculation.
- A frozen BC actor is used as an anchor so PPO does not drift too far from the
  already-working imitation policy.

Policy input stays 27-D: 21-D env observation + 6-D waypoint feature.
Policy output stays 2-D: `[forward_speed_action, steer_action]`.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from dataclasses import dataclass

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


parser = argparse.ArgumentParser(description="PPO fine-tune for waypoint-conditioned FW-mini obstacle navigation.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--bc-checkpoint", type=str, default="logs/fwmini_obstacle_bc_wp/latest.pt")
parser.add_argument("--save-dir", type=str, default="logs/fwmini_obstacle_ppo_bc_wp")

parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--rollout-steps", type=int, default=256)
parser.add_argument("--ppo-epochs", type=int, default=4)
parser.add_argument("--minibatch-size", type=int, default=8192)

parser.add_argument("--actor-lr", type=float, default=5e-5)
parser.add_argument("--critic-lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-eps", type=float, default=0.10)
parser.add_argument("--entropy-coef", type=float, default=0.001)
parser.add_argument("--value-coef", type=float, default=0.5)
parser.add_argument("--bc-coef", type=float, default=0.02)
parser.add_argument("--target-kl", type=float, default=0.05)
parser.add_argument("--max-grad-norm", type=float, default=1.0)

parser.add_argument("--init-log-std", type=float, default=-2.0)
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


def set_seed(seed: int):
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


def extract_reward(rew):
    if isinstance(rew, dict):
        return rew["policy"]
    return rew


class WaypointState:
    def __init__(self, num_envs, device):
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pass_side = torch.ones(num_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids):
        self.phase[env_ids] = 0
        self.pass_side[env_ids] = 1.0


def waypoint_features(env, obs_tensor, state):
    """Compute the same 6-D waypoint feature used during BC pretraining."""

    device = obs_tensor.device
    num_envs = obs_tensor.shape[0]

    origins = env.scene.env_origins
    root_pos = env.robot.data.root_pos_w[:, :3]

    local_x = root_pos[:, 0] - origins[:, 0]
    local_y = root_pos[:, 1] - origins[:, 1]

    obs_cx, obs_cy, obs_sx, obs_sy, _ = env.cfg.obstacles[0]

    goal_local_y = env._goal_w[:, 1] - origins[:, 1]
    new_side = torch.where(goal_local_y >= 0.0, 1.0, -1.0)

    # In phase 0, choose whether the waypoint lane goes above or below the obstacle.
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

    wp = torch.stack(
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

    return wp


class Actor(nn.Module):
    """Policy network. It maps normalized observation to mean action."""

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


class Critic(nn.Module):
    """Value network used by PPO to estimate future return."""

    def __init__(self, obs_dim, hidden_dims=(256, 256)):
        super().__init__()

        layers = []
        last = obs_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h

        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_norm):
        return self.net(obs_norm).squeeze(-1)


def make_dist(mean, log_std):
    std = torch.exp(log_std).expand_as(mean)
    return torch.distributions.Normal(mean, std)


def compute_gae(rewards, dones, values, last_value, gamma, lam):
    """Compute Generalized Advantage Estimation for PPO.

    Shapes:
    - rewards: [T, N]
    - dones: [T, N]
    - values: [T, N]
    - last_value: [N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_nonterminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def save_checkpoint(path, actor, critic, log_std, obs_mean, obs_std, cfg_dict, iteration):
    ckpt = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "log_std": log_std.detach().cpu(),
        "obs_mean": obs_mean.detach().cpu(),
        "obs_std": obs_std.detach().cpu(),
        "obs_dim": cfg_dict["obs_dim"],
        "act_dim": cfg_dict["act_dim"],
        "hidden_dims": cfg_dict["hidden_dims"],
        "iteration": iteration,
        "task": "fwmini_obstacle_ppo_bc_wp",
    }
    torch.save(ckpt, path)


def main():
    """Load BC policy, collect PPO rollouts, update actor/critic, save checkpoints."""

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

    bc_ckpt = torch.load(args_cli.bc_checkpoint, map_location=device)

    obs_mean = bc_ckpt["obs_mean"].to(device)
    obs_std = bc_ckpt["obs_std"].to(device)

    obs_dim = int(bc_ckpt["obs_dim"])
    act_dim = int(bc_ckpt["act_dim"])
    hidden_dims = tuple(bc_ckpt.get("hidden_dims", [256, 256]))

    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims).to(device)
    actor.load_state_dict(bc_ckpt["model"], strict=True)

    # Freeze a copy of the BC actor as a regularization anchor during PPO.
    bc_anchor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims).to(device)
    bc_anchor.load_state_dict(bc_ckpt["model"], strict=True)
    bc_anchor.eval()
    for p in bc_anchor.parameters():
        p.requires_grad_(False)

    critic = Critic(obs_dim=obs_dim, hidden_dims=hidden_dims).to(device)

    log_std = torch.nn.Parameter(
        torch.ones(act_dim, device=device) * float(args_cli.init_log_std)
    )

    optimizer = optim.Adam(
        [
            {"params": actor.parameters(), "lr": args_cli.actor_lr},
            {"params": critic.parameters(), "lr": args_cli.critic_lr},
            {"params": [log_std], "lr": args_cli.actor_lr},
        ]
    )

    state = WaypointState(num_envs=args_cli.num_envs, device=device)

    print("[INFO] Start obstacle PPO fine-tune")
    print(f"[INFO] bc_checkpoint={args_cli.bc_checkpoint}")
    print(f"[INFO] save_dir={args_cli.save_dir}")
    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"[INFO] num_envs={args_cli.num_envs}, rollout_steps={args_cli.rollout_steps}")
    print("[INFO] obstacles:", cfg.obstacles)

    total_episodes = 0
    success = 0
    collision = 0
    timeout = 0
    bad_z = 0
    bad_wheel = 0
    bad_steer = 0
    unknown_failure = 0

    for it in range(1, args_cli.iterations + 1):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        mean_buf = []

        ep_return_sum = 0.0

        for step in range(args_cli.rollout_steps):
            # Rollout phase: sample actions from the current actor in IsaacLab.
            with torch.no_grad():
                wp = waypoint_features(env, obs_tensor, state)
                obs_aug = torch.cat([obs_tensor, wp], dim=1)
                obs_norm = (obs_aug - obs_mean) / obs_std

                mean = actor(obs_norm)
                value = critic(obs_norm)

                dist = make_dist(mean, log_std)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action).sum(dim=-1)
                action = torch.clamp(raw_action, -1.0, 1.0)

            warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
            action[warmup_mask, :] = 0.0

            obs_next, rew, terminated, truncated, info = env.step(action)

            reward = extract_reward(rew).reshape(-1)
            obs_next_tensor = extract_policy_obs(obs_next)

            done = (terminated | truncated).reshape(-1)

            obs_buf.append(obs_aug.detach())
            act_buf.append(raw_action.detach())
            logp_buf.append(log_prob.detach())
            rew_buf.append(reward.detach())
            done_buf.append(done.float().detach())
            val_buf.append(value.detach())
            mean_buf.append(mean.detach())

            ep_return_sum += float(reward.mean().item())

            done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

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

            obs_tensor = obs_next_tensor

        with torch.no_grad():
            wp = waypoint_features(env, obs_tensor, state)
            obs_aug = torch.cat([obs_tensor, wp], dim=1)
            obs_norm = (obs_aug - obs_mean) / obs_std
            last_value = critic(obs_norm)

        obs_batch = torch.stack(obs_buf, dim=0)
        act_batch = torch.stack(act_buf, dim=0)
        old_logp_batch = torch.stack(logp_buf, dim=0)
        rewards = torch.stack(rew_buf, dim=0)
        dones = torch.stack(done_buf, dim=0)
        values = torch.stack(val_buf, dim=0)
        old_mean_batch = torch.stack(mean_buf, dim=0)

        # Convert rewards/values into PPO training targets.
        advantages, returns = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            last_value=last_value,
            gamma=args_cli.gamma,
            lam=args_cli.gae_lambda,
        )

        T, N = rewards.shape
        batch_size = T * N

        obs_flat = obs_batch.reshape(batch_size, obs_dim)
        act_flat = act_batch.reshape(batch_size, act_dim)
        old_logp_flat = old_logp_batch.reshape(batch_size)
        adv_flat = advantages.reshape(batch_size)
        ret_flat = returns.reshape(batch_size)
        val_flat = values.reshape(batch_size)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        approx_kl_value = 0.0
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        bc_loss_value = 0.0

        idx_all = torch.arange(batch_size, device=device)

        for epoch in range(args_cli.ppo_epochs):
            # Optimization phase: clipped PPO loss + value loss + BC anchor loss.
            perm = idx_all[torch.randperm(batch_size, device=device)]

            for start in range(0, batch_size, args_cli.minibatch_size):
                idx = perm[start : start + args_cli.minibatch_size]

                mb_obs = obs_flat[idx]
                mb_obs_norm = (mb_obs - obs_mean) / obs_std

                mb_act = act_flat[idx]
                mb_old_logp = old_logp_flat[idx]
                mb_adv = adv_flat[idx]
                mb_ret = ret_flat[idx]

                mean = actor(mb_obs_norm)
                value = critic(mb_obs_norm)

                dist = make_dist(mean, log_std)
                logp = dist.log_prob(mb_act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(logp - mb_old_logp)

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - args_cli.clip_eps, 1.0 + args_cli.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                value_loss = torch.mean((value - mb_ret) ** 2)

                with torch.no_grad():
                    bc_mean = bc_anchor(mb_obs_norm)

                bc_loss = torch.mean((mean - bc_mean) ** 2)

                loss = (
                    policy_loss
                    + args_cli.value_coef * value_loss
                    - args_cli.entropy_coef * entropy
                    + args_cli.bc_coef * bc_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()) + [log_std],
                    args_cli.max_grad_norm,
                )
                optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - logp).abs()

                approx_kl_value = float(approx_kl.item())
                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                entropy_value = float(entropy.item())
                bc_loss_value = float(bc_loss.item())

            if approx_kl_value > args_cli.target_kl:
                break

        failure = bad_z + bad_wheel + bad_steer + unknown_failure
        success_rate = success / max(total_episodes, 1)
        collision_rate = collision / max(total_episodes, 1)
        timeout_rate = timeout / max(total_episodes, 1)
        failure_rate = failure / max(total_episodes, 1)

        mean_return = rewards.sum(dim=0).mean().item()
        mean_reward = rewards.mean().item()

        if it % args_cli.print_every == 0 or it == 1:
            print(
                f"[ITER {it:04d}] "
                f"episodes={total_episodes} | "
                f"success_rate={success_rate:.3f} | "
                f"collision_rate={collision_rate:.3f} | "
                f"timeout_rate={timeout_rate:.3f} | "
                f"failure_rate={failure_rate:.3f} | "
                f"mean_return={mean_return:.3f} | "
                f"mean_reward={mean_reward:.4f} | "
                f"policy_loss={policy_loss_value:.5f} | "
                f"value_loss={value_loss_value:.5f} | "
                f"entropy={entropy_value:.5f} | "
                f"bc_loss={bc_loss_value:.6f} | "
                f"kl={approx_kl_value:.5f} | "
                f"log_std={log_std.mean().item():.3f}"
            )

        if it % 25 == 0 or it == args_cli.iterations:
            ckpt_cfg = {
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_dims": list(hidden_dims),
            }

            iter_path = os.path.join(args_cli.save_dir, f"iter_{it:04d}.pt")
            latest_path = os.path.join(args_cli.save_dir, "latest.pt")

            save_checkpoint(
                iter_path,
                actor,
                critic,
                log_std,
                obs_mean,
                obs_std,
                ckpt_cfg,
                it,
            )
            save_checkpoint(
                latest_path,
                actor,
                critic,
                log_std,
                obs_mean,
                obs_std,
                ckpt_cfg,
                it,
            )

            print(f"[INFO] Saved checkpoint: {latest_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
