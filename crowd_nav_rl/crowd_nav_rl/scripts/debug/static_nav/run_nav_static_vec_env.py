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


parser = argparse.ArgumentParser(description="Run vectorized FW-mini random-goal nav_static smoke test.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--print-every", type=int, default=100)
parser.add_argument("--num-envs", type=int, default=8)
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


def policy_from_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
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


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavStaticVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs

    env = FWMiniNavStaticVecEnv(cfg=cfg)
    obs, _ = env.reset()

    print("[INFO] vectorized random-goal nav_static env reset done")
    print(f"[INFO] seed={args_cli.seed}, num_envs={args_cli.num_envs}")
    print("[INFO] obs shape:", obs["policy"].shape)

    total_episodes = 0
    success_count = 0
    timeout_count = 0
    failure_count = 0

    for i in range(args_cli.steps):
        obs_tensor = obs["policy"]
        actions = policy_from_obs(obs_tensor)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        actions[warmup_mask, :] = 0.0

        obs, rew, terminated, truncated, info = env.step(actions)

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if i % args_cli.print_every == 0:
            root_pos = env.robot.data.root_pos_w[0, :3]
            root_vel = env.robot.data.root_lin_vel_w[0, :3]
            obs0 = obs["policy"][0]

            print(
                f"step {i} | "
                f"env0_pos=({root_pos[0].item():.4f}, {root_pos[1].item():.4f}, {root_pos[2].item():.4f}) | "
                f"env0_vel=({root_vel[0].item():.4f}, {root_vel[1].item():.4f}, {root_vel[2].item():.4f}) | "
                f"env0_dist={obs0[4].item():.4f} | "
                f"env0_heading={obs0[5].item():.4f} | "
                f"mean_dist={obs['policy'][:, 4].mean().item():.4f} | "
                f"done_this_step={len(done_ids)} | "
                f"episodes={total_episodes} | "
                f"success={success_count} | "
                f"timeout={timeout_count} | "
                f"failure={failure_count}"
            )

        if len(done_ids) > 0:
            reasons = env.done_reason_buf[done_ids].detach().cpu().tolist()

            for rid, reason_code in zip(done_ids.detach().cpu().tolist(), reasons):
                reason = REASON_MAP.get(int(reason_code), "unknown")

                total_episodes += 1

                if reason == "goal":
                    success_count += 1
                elif reason == "timeout":
                    timeout_count += 1
                else:
                    failure_count += 1

                success_rate = success_count / max(total_episodes, 1)

                print(
                    f"[EPISODE END] global_step={i} | env_id={rid} | "
                    f"reason={reason} | "
                    f"goal_dist={env.last_goal_dist_buf[rid].item():.4f} | "
                    f"z_drop={env.last_z_drop_buf[rid].item():.5f} | "
                    f"max_wheel={env.last_max_wheel_buf[rid].item():.4f} | "
                    f"max_steer={env.last_max_steer_buf[rid].item():.4f} | "
                    f"episodes={total_episodes} | "
                    f"success={success_count} | "
                    f"timeout={timeout_count} | "
                    f"failure={failure_count} | "
                    f"success_rate={success_rate:.3f}"
                )

    print("\n================ VEC SUMMARY ================")
    print(f"num_envs     : {args_cli.num_envs}")
    print(f"steps        : {args_cli.steps}")
    print(f"episodes     : {total_episodes}")
    print(f"success      : {success_count}")
    print(f"timeout      : {timeout_count}")
    print(f"failure      : {failure_count}")
    if total_episodes > 0:
        print(f"success_rate : {success_count / total_episodes:.3f}")
    print("============================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()