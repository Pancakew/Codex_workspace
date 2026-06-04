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


parser = argparse.ArgumentParser(description="Behavior cloning pretraining for FW-mini static navigation.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--collect-steps", type=int, default=4000)
parser.add_argument("--train-epochs", type=int, default=80)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--save-dir", type=str, default="logs/fwmini_static_bc")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--print-every", type=int, default=10)

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


def expert_policy_from_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
    """
    Hand-crafted expert policy.

    obs:
        0 last_v_cmd
        1 last_steer_cmd
        2 goal_body_x
        3 goal_body_y
        4 goal_dist
        5 heading_error
    """
    goal_dist = obs_tensor[:, 4]
    heading_error = obs_tensor[:, 5]

    abs_heading = torch.abs(heading_error)

    v_action = torch.where(
        abs_heading > 1.0,
        torch.full_like(abs_heading, 0.20),
        torch.where(
            abs_heading > 0.5,
            torch.full_like(abs_heading, 0.40),
            torch.full_like(abs_heading, 0.80),
        ),
    )

    v_action = torch.where(goal_dist < 0.35, torch.zeros_like(v_action), v_action)

    steer_action = torch.clamp(heading_error / 0.25, -1.0, 1.0)

    return torch.stack([v_action, steer_action], dim=1)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(128, 128)):
        super().__init__()

        layers = []
        last_dim = obs_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h

        layers.append(nn.Linear(last_dim, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs_norm))


def main():
    set_seed(args_cli.seed)
    os.makedirs(args_cli.save_dir, exist_ok=True)

    cfg = FWMiniNavStaticVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs

    env = FWMiniNavStaticVecEnv(cfg=cfg)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    device = env.device
    obs_dim = obs.shape[1]
    act_dim = cfg.action_space

    print("[INFO] Start collecting expert data")
    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}, num_envs={args_cli.num_envs}")

    obs_chunks = []
    act_chunks = []

    episode_count = 0
    success_count = 0
    timeout_count = 0
    failure_count = 0

    for step in range(args_cli.collect_steps):
        with torch.no_grad():
            expert_action = expert_policy_from_obs(obs)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        expert_action[warmup_mask, :] = 0.0

        valid_mask = ~warmup_mask

        if torch.any(valid_mask):
            obs_chunks.append(obs[valid_mask].detach().clone())
            act_chunks.append(expert_action[valid_mask].detach().clone())

        next_obs_dict, rew, terminated, truncated, info = env.step(expert_action)
        obs = next_obs_dict["policy"]

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if len(done_ids) > 0:
            reasons = env.done_reason_buf[done_ids].detach().cpu().tolist()

            for r in reasons:
                episode_count += 1
                if int(r) == 1:
                    success_count += 1
                elif int(r) == 5:
                    timeout_count += 1
                else:
                    failure_count += 1

        if step % 500 == 0:
            success_rate = success_count / max(episode_count, 1)
            print(
                f"[COLLECT] step={step} | "
                f"samples={sum(x.shape[0] for x in obs_chunks)} | "
                f"episodes={episode_count} | "
                f"success={success_count} | "
                f"timeout={timeout_count} | "
                f"failure={failure_count} | "
                f"success_rate={success_rate:.3f} | "
                f"mean_dist={obs[:, 4].mean().item():.4f}"
            )

    dataset_obs = torch.cat(obs_chunks, dim=0)
    dataset_act = torch.cat(act_chunks, dim=0)

    print("[INFO] Expert data collected")
    print(f"[INFO] dataset_obs={dataset_obs.shape}, dataset_act={dataset_act.shape}")

    obs_mean = dataset_obs.mean(dim=0, keepdim=True)
    obs_std = dataset_obs.std(dim=0, keepdim=True).clamp_min(1e-6)

    dataset_obs_norm = (dataset_obs - obs_mean) / obs_std

    model = BCPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=(128, 128)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args_cli.lr)

    dataset_size = dataset_obs_norm.shape[0]

    print("[INFO] Start behavior cloning training")

    for epoch in range(args_cli.train_epochs):
        perm = torch.randperm(dataset_size, device=device)

        total_loss = 0.0
        total_batches = 0

        for start in range(0, dataset_size, args_cli.batch_size):
            idx = perm[start : start + args_cli.batch_size]

            batch_obs = dataset_obs_norm[idx]
            batch_act = dataset_act[idx]

            pred_act = model(batch_obs)
            loss = torch.mean((pred_act - batch_act) ** 2)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        mean_loss = total_loss / max(total_batches, 1)

        if epoch % args_cli.print_every == 0 or epoch == args_cli.train_epochs - 1:
            with torch.no_grad():
                pred = model(dataset_obs_norm[: min(4096, dataset_size)])
                target = dataset_act[: min(4096, dataset_size)]
                eval_mse = torch.mean((pred - target) ** 2).item()
                eval_mae = torch.mean(torch.abs(pred - target)).item()

            print(
                f"[BC EPOCH {epoch + 1:04d}] "
                f"train_loss={mean_loss:.6f} | "
                f"eval_mse={eval_mse:.6f} | "
                f"eval_mae={eval_mae:.6f}"
            )

    checkpoint = {
        "model": model.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dims": [128, 128],
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "collect_steps": args_cli.collect_steps,
        "train_epochs": args_cli.train_epochs,
    }

    latest_path = os.path.join(args_cli.save_dir, "latest.pt")
    policy_path = os.path.join(args_cli.save_dir, "bc_policy.pt")

    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, policy_path)

    print(f"[INFO] Saved BC checkpoint: {latest_path}")
    print(f"[INFO] Saved BC checkpoint: {policy_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()