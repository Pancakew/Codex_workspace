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


parser = argparse.ArgumentParser(description="Play behavior-cloned FW-mini static navigation policy.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--checkpoint", type=str, default="logs/fwmini_static_bc/latest.pt")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--print-every", type=int, default=100)
parser.add_argument("--seed", type=int, default=123)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from crowd_nav_rl.tasks.locomotion.nav_static_vec.fwmini_nav_static_vec_env_cfg import (
    FWMiniNavStaticVecEnvCfg,
)
from crowd_nav_rl.tasks.locomotion.nav_static_vec.fwmini_nav_static_vec_env import (
    FWMiniNavStaticVecEnv,
)


REASON_MAP = {
    0: "none",
    1: "goal",
    2: "bad_z",
    3: "bad_wheel",
    4: "bad_steer",
    5: "timeout",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    cfg = FWMiniNavStaticVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs

    env = FWMiniNavStaticVecEnv(cfg=cfg)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    device = env.device

    ckpt = torch.load(args_cli.checkpoint, map_location=device)

    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    hidden_dims = tuple(ckpt.get("hidden_dims", [128, 128]))

    model = BCPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)

    total_episodes = 0
    success = 0
    timeout = 0
    failure = 0
    bad_z = 0
    bad_wheel = 0
    bad_steer = 0

    print("[INFO] Start BC policy play")
    print(f"[INFO] checkpoint={args_cli.checkpoint}")
    print(f"[INFO] num_envs={args_cli.num_envs}, seed={args_cli.seed}")

    for step in range(args_cli.steps):
        with torch.no_grad():
            obs_norm = (obs - obs_mean) / obs_std
            action = model(obs_norm)
            action = torch.clamp(action, -1.0, 1.0)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        action[warmup_mask, :] = 0.0

        obs_dict, rew, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if step % args_cli.print_every == 0:
            print(
                f"step={step} | "
                f"mean_dist={obs[:, 4].mean().item():.4f} | "
                f"min_dist={obs[:, 4].min().item():.4f} | "
                f"max_dist={obs[:, 4].max().item():.4f} | "
                f"mean_action=({action[:, 0].mean().item():.3f}, {action[:, 1].mean().item():.3f}) | "
                f"episodes={total_episodes} | "
                f"success={success} | "
                f"timeout={timeout} | "
                f"failure={failure}"
            )

        if len(done_ids) > 0:
            reason_codes = env.done_reason_buf[done_ids].detach().cpu().tolist()

            for reason_code in reason_codes:
                reason = REASON_MAP.get(int(reason_code), "unknown")

                total_episodes += 1

                if reason == "goal":
                    success += 1
                elif reason == "timeout":
                    timeout += 1
                else:
                    failure += 1

                    if reason == "bad_z":
                        bad_z += 1
                    elif reason == "bad_wheel":
                        bad_wheel += 1
                    elif reason == "bad_steer":
                        bad_steer += 1

    print("\n================ BC PLAY SUMMARY ================")
    print(f"checkpoint   : {args_cli.checkpoint}")
    print(f"num_envs     : {args_cli.num_envs}")
    print(f"steps        : {args_cli.steps}")
    print(f"episodes     : {total_episodes}")
    print(f"success      : {success}")
    print(f"timeout      : {timeout}")
    print(f"failure      : {failure}")
    print(f"bad_z        : {bad_z}")
    print(f"bad_wheel    : {bad_wheel}")
    print(f"bad_steer    : {bad_steer}")
    if total_episodes > 0:
        print(f"success_rate : {success / total_episodes:.3f}")
    print("=================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()