import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play FW-mini static navigation policy.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--checkpoint", type=str, default="logs/fwmini_static_ppo/latest.pt")
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


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(128, 128), activation="tanh"):
        super().__init__()

        act_layer = nn.Tanh if activation == "tanh" else nn.ReLU

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


def main():
    set_seed(args_cli.seed)

    ckpt = torch.load(args_cli.checkpoint, map_location="cuda:0")
    cfg_dict = ckpt["config"]

    env_cfg = FWMiniNavStaticVecEnvCfg()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs

    env = FWMiniNavStaticVecEnv(cfg=env_cfg)
    obs_dict, _ = env.reset()

    obs = obs_dict["policy"]
    obs_dim = obs.shape[1]
    act_dim = env_cfg.action_space

    net_cfg = cfg_dict["network"]

    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=tuple(net_cfg.get("hidden_dims", [128, 128])),
        activation=net_cfg.get("activation", "tanh"),
    ).to(env.device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    total_episodes = 0
    success = 0
    timeout = 0
    failure = 0

    print("[INFO] Start play")
    print(f"[INFO] checkpoint={args_cli.checkpoint}")
    print(f"[INFO] num_envs={args_cli.num_envs}")

    for i in range(args_cli.steps):
        with torch.no_grad():
            mean, _ = model(obs)
            action = torch.clamp(mean, -1.0, 1.0)

        # Force zero action during each episode warmup period.
        warmup_mask = env.episode_length_buf < (env_cfg.warmup_steps + 20)
        action[warmup_mask, :] = 0.0

        obs_dict, rew, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if i % args_cli.print_every == 0:
            print(
                f"step={i} | "
                f"mean_dist={obs[:, 4].mean().item():.4f} | "
                f"min_dist={obs[:, 4].min().item():.4f} | "
                f"max_dist={obs[:, 4].max().item():.4f} | "
                f"episodes={total_episodes} | "
                f"success={success} | timeout={timeout} | failure={failure}"
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

    print("\n================ PLAY SUMMARY ================")
    print(f"checkpoint   : {args_cli.checkpoint}")
    print(f"num_envs     : {args_cli.num_envs}")
    print(f"steps        : {args_cli.steps}")
    print(f"episodes     : {total_episodes}")
    print(f"success      : {success}")
    print(f"timeout      : {timeout}")
    print(f"failure      : {failure}")
    if total_episodes > 0:
        print(f"success_rate : {success / total_episodes:.3f}")
    print("==============================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
