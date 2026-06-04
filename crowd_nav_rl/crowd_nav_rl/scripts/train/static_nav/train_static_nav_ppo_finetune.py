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


parser = argparse.ArgumentParser(description="BC-initialized PPO fine-tuning for FW-mini static navigation.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--bc-checkpoint", type=str, default="logs/fwmini_static_bc/latest.pt")
parser.add_argument("--save-dir", type=str, default="logs/fwmini_static_ppo_bc")
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--max-iters", type=int, default=200)
parser.add_argument("--horizon", type=int, default=64)
parser.add_argument("--minibatch-size", type=int, default=8192)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-param", type=float, default=0.10)
parser.add_argument("--value-loss-coef", type=float, default=1.0)
parser.add_argument("--entropy-coef", type=float, default=0.001)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--target-kl", type=float, default=0.01)
parser.add_argument("--critic-warmup-iters", type=int, default=20)
parser.add_argument("--save-interval", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from crowd_nav_rl.tasks.locomotion.nav_static_vec.fwmini_nav_static_vec_env_cfg import (
    FWMiniNavStaticVecEnvCfg,
)
from crowd_nav_rl.tasks.locomotion.nav_static_vec.fwmini_nav_static_vec_env import (
    FWMiniNavStaticVecEnv,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def build_mlp(input_dim: int, output_dim: int, hidden_dims=(128, 128)):
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.Tanh())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Actor:
        obs_norm -> raw_action_mean
        action = tanh(raw_action)

    Critic:
        obs_norm -> value
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(128, 128)):
        super().__init__()

        self.actor_net = build_mlp(obs_dim, act_dim, hidden_dims)
        self.critic_net = build_mlp(obs_dim, 1, hidden_dims)

        # Use small exploration noise so PPO does not immediately destroy the BC policy.
        self.log_std = nn.Parameter(torch.ones(act_dim) * -2.0)

    def get_mean_action(self, obs_norm: torch.Tensor) -> torch.Tensor:
        raw_mean = self.actor_net(obs_norm)
        return torch.tanh(raw_mean)

    def get_value(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return self.critic_net(obs_norm).squeeze(-1)

    def act(self, obs_norm: torch.Tensor):
        raw_mean = self.actor_net(obs_norm)
        std = torch.exp(self.log_std).expand_as(raw_mean)

        dist = torch.distributions.Normal(raw_mean, std)

        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - action * action + 1e-6).sum(dim=-1)
        log_prob = log_prob_raw - correction

        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(obs_norm)

        return action, log_prob, entropy, value

    def evaluate_actions(self, obs_norm: torch.Tensor, actions: torch.Tensor):
        raw_mean = self.actor_net(obs_norm)
        std = torch.exp(self.log_std).expand_as(raw_mean)

        dist = torch.distributions.Normal(raw_mean, std)

        raw_action = atanh(actions)

        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - actions * actions + 1e-6).sum(dim=-1)
        log_prob = log_prob_raw - correction

        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(obs_norm)

        return log_prob, entropy, value


def load_bc_into_actor(model: ActorCritic, bc_ckpt: dict):
    """
    BC checkpoint 涓殑 key 鏄?
        net.0.weight, net.0.bias, ...

    PPO actor_net 鐨?key 鏄?
        0.weight, 0.bias, ...

    Therefore this loader removes the `net.` prefix before loading into actor_net.
    """
    bc_state = bc_ckpt["model"]

    actor_state = {}
    for k, v in bc_state.items():
        if k.startswith("net."):
            actor_state[k.replace("net.", "")] = v
        else:
            actor_state[k] = v

    model.actor_net.load_state_dict(actor_state, strict=True)


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    horizon, num_envs = rewards.shape

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        not_done = 1.0 - dones[t].float()

        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns


def main():
    set_seed(args_cli.seed)
    os.makedirs(args_cli.save_dir, exist_ok=True)

    env_cfg = FWMiniNavStaticVecEnvCfg()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs

    env = FWMiniNavStaticVecEnv(cfg=env_cfg)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    device = env.device
    obs_dim = obs.shape[1]
    act_dim = env_cfg.action_space

    bc_ckpt = torch.load(args_cli.bc_checkpoint, map_location=device)

    obs_mean = bc_ckpt["obs_mean"].to(device)
    obs_std = bc_ckpt["obs_std"].to(device)
    hidden_dims = tuple(bc_ckpt.get("hidden_dims", [128, 128]))

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    load_bc_into_actor(model, bc_ckpt)

    optimizer = optim.Adam(model.parameters(), lr=args_cli.lr)

    print("[INFO] Start BC-initialized PPO fine-tuning")
    print(f"[INFO] bc_checkpoint={args_cli.bc_checkpoint}")
    print(f"[INFO] save_dir={args_cli.save_dir}")
    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}, num_envs={args_cli.num_envs}")
    print(f"[INFO] critic_warmup_iters={args_cli.critic_warmup_iters}")

    episode_returns = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device)

    completed_returns = []
    completed_lengths = []

    total_episodes = 0
    success = 0
    timeout = 0
    failure = 0

    for it in range(args_cli.max_iters):
        obs_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, obs_dim), dtype=torch.float32, device=device)
        act_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, act_dim), dtype=torch.float32, device=device)
        logp_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), dtype=torch.float32, device=device)
        rew_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), dtype=torch.float32, device=device)
        done_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), dtype=torch.float32, device=device)
        val_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), dtype=torch.float32, device=device)

        for t in range(args_cli.horizon):
            obs_norm = (obs - obs_mean) / obs_std

            with torch.no_grad():
                action, log_prob, entropy, value = model.act(obs_norm)

            # Keep zero action during each episode warmup period.
            warmup_mask = env.episode_length_buf < (env_cfg.warmup_steps + 20)
            action[warmup_mask, :] = 0.0

            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs_dict["policy"]

            done = terminated | truncated

            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = log_prob
            rew_buf[t] = reward
            done_buf[t] = done.float()
            val_buf[t] = value

            episode_returns += reward
            episode_lengths += 1.0

            done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

            if len(done_ids) > 0:
                for env_id in done_ids.detach().cpu().tolist():
                    total_episodes += 1
                    completed_returns.append(float(episode_returns[env_id].item()))
                    completed_lengths.append(float(episode_lengths[env_id].item()))

                    reason_code = int(env.done_reason_buf[env_id].item())

                    if reason_code == 1:
                        success += 1
                    elif reason_code == 5:
                        timeout += 1
                    else:
                        failure += 1

                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0.0

            obs = next_obs

        with torch.no_grad():
            last_obs_norm = (obs - obs_mean) / obs_std
            last_value = model.get_value(last_obs_norm)

        advantages, returns = compute_gae(
            rewards=rew_buf,
            dones=done_buf,
            values=val_buf,
            last_value=last_value,
            gamma=args_cli.gamma,
            gae_lambda=args_cli.gae_lambda,
        )

        batch_obs = obs_buf.reshape(-1, obs_dim)
        batch_actions = act_buf.reshape(-1, act_dim)
        batch_old_logp = logp_buf.reshape(-1)
        batch_adv = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)

        batch_obs_norm = (batch_obs - obs_mean) / obs_std

        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

        batch_size = batch_obs_norm.shape[0]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        update_count = 0

        actor_enabled = it >= args_cli.critic_warmup_iters

        stop_update = False

        for epoch in range(args_cli.epochs):
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, args_cli.minibatch_size):
                mb_idx = indices[start : start + args_cli.minibatch_size]

                mb_obs = batch_obs_norm[mb_idx]
                mb_actions = batch_actions[mb_idx]
                mb_old_logp = batch_old_logp[mb_idx]
                mb_adv = batch_adv[mb_idx]
                mb_returns = batch_returns[mb_idx]

                new_logp, entropy, values = model.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)

                unclipped = ratio * mb_adv
                clipped = torch.clamp(
                    ratio,
                    1.0 - args_cli.clip_param,
                    1.0 + args_cli.clip_param,
                ) * mb_adv

                policy_loss = -torch.mean(torch.min(unclipped, clipped))
                value_loss = torch.mean((mb_returns - values) ** 2)
                entropy_bonus = torch.mean(entropy)

                if actor_enabled:
                    loss = (
                        policy_loss
                        + args_cli.value_loss_coef * value_loss
                        - args_cli.entropy_coef * entropy_bonus
                    )
                else:
                    # For the first few iterations, train only the critic.
                    loss = args_cli.value_loss_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args_cli.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - new_logp).abs()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_bonus.item())
                total_kl += float(approx_kl.item())
                update_count += 1

                if actor_enabled and approx_kl > args_cli.target_kl:
                    stop_update = True
                    break

            if stop_update:
                break

        mean_return = float(np.mean(completed_returns[-200:])) if completed_returns else 0.0
        mean_length = float(np.mean(completed_lengths[-200:])) if completed_lengths else 0.0
        success_rate = success / max(total_episodes, 1)
        timeout_rate = timeout / max(total_episodes, 1)
        failure_rate = failure / max(total_episodes, 1)

        print(
            f"[ITER {it + 1:04d}] "
            f"actor_update={actor_enabled} | "
            f"episodes={total_episodes} | "
            f"success_rate={success_rate:.3f} | "
            f"timeout_rate={timeout_rate:.3f} | "
            f"failure_rate={failure_rate:.3f} | "
            f"mean_return={mean_return:.3f} | "
            f"mean_len={mean_length:.1f} | "
            f"policy_loss={total_policy_loss / max(update_count, 1):.5f} | "
            f"value_loss={total_value_loss / max(update_count, 1):.5f} | "
            f"entropy={total_entropy / max(update_count, 1):.5f} | "
            f"kl={total_kl / max(update_count, 1):.5f} | "
            f"log_std={model.log_std.detach().mean().item():.3f}"
        )

        if (it + 1) % args_cli.save_interval == 0 or it == args_cli.max_iters - 1:
            ckpt = {
                "model": model.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_dims": list(hidden_dims),
                "iteration": it + 1,
                "bc_checkpoint": args_cli.bc_checkpoint,
            }

            latest_path = os.path.join(args_cli.save_dir, "latest.pt")
            iter_path = os.path.join(args_cli.save_dir, f"iter_{it + 1:04d}.pt")

            torch.save(ckpt, latest_path)
            torch.save(ckpt, iter_path)

            print(f"[INFO] Saved checkpoint: {latest_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
