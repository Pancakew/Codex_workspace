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
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train FW-mini static navigation with PPO.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument(
    "--config",
    type=str,
    default="crowd_nav_rl/tasks/agents/rl_games_ppo_static.yaml",
)
parser.add_argument("--checkpoint", type=str, default=None)

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


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(128, 128), activation="tanh"):
        super().__init__()

        if activation == "relu":
            act_layer = nn.ReLU
        else:
            act_layer = nn.Tanh

        actor_layers = []
        last_dim = obs_dim
        for h in hidden_dims:
            actor_layers.append(nn.Linear(last_dim, h))
            actor_layers.append(act_layer())
            last_dim = h
        actor_layers.append(nn.Linear(last_dim, act_dim))

        critic_layers = []
        last_dim = obs_dim
        for h in hidden_dims:
            critic_layers.append(nn.Linear(last_dim, h))
            critic_layers.append(act_layer())
            last_dim = h
        critic_layers.append(nn.Linear(last_dim, 1))

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return mean, value

    def act(self, obs: torch.Tensor):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.clamp(raw_action, -1.0, 1.0)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        # This smoke PPO uses clipped actions directly for log-prob estimation.
        # A later production version should use a tanh-squashed Gaussian.
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    horizon, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_adv = delta + gamma * gae_lambda * not_done * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def main():
    cfg_dict = load_yaml(args_cli.config)

    seed = int(cfg_dict.get("seed", 42))
    set_seed(seed)

    num_envs = int(cfg_dict["env"]["num_envs"])
    episode_length_s = float(cfg_dict["env"].get("episode_length_s", 40.0))

    ppo_cfg = cfg_dict["ppo"]
    net_cfg = cfg_dict["network"]
    save_cfg = cfg_dict["save"]

    log_dir = save_cfg.get("log_dir", "logs/fwmini_static_ppo")
    os.makedirs(log_dir, exist_ok=True)

    env_cfg = FWMiniNavStaticVecEnvCfg()
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = episode_length_s

    env = FWMiniNavStaticVecEnv(cfg=env_cfg)
    obs_dict, _ = env.reset()

    obs = obs_dict["policy"]
    obs_dim = obs.shape[1]
    act_dim = env_cfg.action_space

    device = env.device

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=tuple(net_cfg.get("hidden_dims", [128, 128])),
        activation=net_cfg.get("activation", "tanh"),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(ppo_cfg["learning_rate"]))

    start_iter = 0
    if args_cli.checkpoint is not None:
        ckpt = torch.load(args_cli.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt.get("iteration", 0))
        print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}, iteration={start_iter}")

    max_iterations = int(ppo_cfg["max_iterations"])
    horizon_length = int(ppo_cfg["horizon_length"])
    minibatch_size = int(ppo_cfg["minibatch_size"])
    learning_epochs = int(ppo_cfg["learning_epochs"])

    gamma = float(ppo_cfg["gamma"])
    gae_lambda = float(ppo_cfg["gae_lambda"])
    clip_param = float(ppo_cfg["clip_param"])
    value_loss_coef = float(ppo_cfg["value_loss_coef"])
    entropy_coef = float(ppo_cfg["entropy_coef"])
    max_grad_norm = float(ppo_cfg["max_grad_norm"])
    target_kl = float(ppo_cfg.get("target_kl", 0.02))
    save_interval = int(save_cfg.get("save_interval", 25))

    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.float32, device=device)

    completed_returns = []
    completed_lengths = []
    completed_success = 0
    completed_total = 0

    print("[INFO] Start PPO training")
    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}, num_envs={num_envs}")

    for it in range(start_iter, max_iterations):
        obs_buf = torch.zeros((horizon_length, num_envs, obs_dim), dtype=torch.float32, device=device)
        act_buf = torch.zeros((horizon_length, num_envs, act_dim), dtype=torch.float32, device=device)
        logp_buf = torch.zeros((horizon_length, num_envs), dtype=torch.float32, device=device)
        rew_buf = torch.zeros((horizon_length, num_envs), dtype=torch.float32, device=device)
        done_buf = torch.zeros((horizon_length, num_envs), dtype=torch.float32, device=device)
        val_buf = torch.zeros((horizon_length, num_envs), dtype=torch.float32, device=device)

        for t in range(horizon_length):
            with torch.no_grad():
                action, log_prob, entropy, value = model.act(obs)

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
                    completed_returns.append(float(episode_returns[env_id].item()))
                    completed_lengths.append(float(episode_lengths[env_id].item()))

                    reason_code = int(env.done_reason_buf[env_id].item())
                    if reason_code == 1:
                        completed_success += 1
                    completed_total += 1

                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0.0

            obs = next_obs

        with torch.no_grad():
            _, last_value = model.forward(obs)

        advantages, returns = compute_gae(
            rewards=rew_buf,
            dones=done_buf,
            values=val_buf,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        batch_obs = obs_buf.reshape(-1, obs_dim)
        batch_actions = act_buf.reshape(-1, act_dim)
        batch_old_logp = logp_buf.reshape(-1)
        batch_adv = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_old_values = val_buf.reshape(-1)

        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

        batch_size = batch_obs.shape[0]
        indices = torch.randperm(batch_size, device=device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        update_count = 0

        stop_update = False

        for epoch in range(learning_epochs):
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                mb_obs = batch_obs[mb_idx]
                mb_actions = batch_actions[mb_idx]
                mb_old_logp = batch_old_logp[mb_idx]
                mb_adv = batch_adv[mb_idx]
                mb_returns = batch_returns[mb_idx]

                new_logp, entropy, values = model.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                value_loss = torch.mean((mb_returns - values) ** 2)
                entropy_loss = torch.mean(entropy)

                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - new_logp).abs()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_loss.item())
                total_kl += float(approx_kl.item())
                update_count += 1

                if approx_kl > target_kl:
                    stop_update = True
                    break

            if stop_update:
                break

        mean_return = float(np.mean(completed_returns[-100:])) if completed_returns else 0.0
        mean_length = float(np.mean(completed_lengths[-100:])) if completed_lengths else 0.0
        success_rate = completed_success / max(completed_total, 1)

        print(
            f"[ITER {it + 1:05d}] "
            f"mean_return={mean_return:.3f} | "
            f"mean_len={mean_length:.1f} | "
            f"success_rate={success_rate:.3f} | "
            f"episodes={completed_total} | "
            f"policy_loss={total_policy_loss / max(update_count, 1):.5f} | "
            f"value_loss={total_value_loss / max(update_count, 1):.5f} | "
            f"entropy={total_entropy / max(update_count, 1):.5f} | "
            f"kl={total_kl / max(update_count, 1):.5f}"
        )

        if (it + 1) % save_interval == 0 or it == max_iterations - 1:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": it + 1,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "config": cfg_dict,
            }

            latest_path = os.path.join(log_dir, "latest.pt")
            iter_path = os.path.join(log_dir, f"iter_{it + 1:05d}.pt")

            torch.save(ckpt, latest_path)
            torch.save(ckpt, iter_path)

            print(f"[INFO] Saved checkpoint: {latest_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
