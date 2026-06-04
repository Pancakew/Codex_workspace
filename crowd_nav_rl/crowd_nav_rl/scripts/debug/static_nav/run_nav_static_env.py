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


parser = argparse.ArgumentParser(description="Run FW-mini random-goal static navigation smoke test.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from crowd_nav_rl.tasks.locomotion.nav_static.fwmini_nav_static_env_cfg import (
    FWMiniNavStaticEnvCfg,
)
from crowd_nav_rl.tasks.locomotion.nav_static.fwmini_nav_static_env import (
    FWMiniNavStaticEnv,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def policy_from_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
    goal_dist = obs_tensor[0, 4]
    heading_error = obs_tensor[0, 5]

    if goal_dist < 0.35:
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=obs_tensor.device)

    abs_heading = torch.abs(heading_error)

    if abs_heading > 1.0:
        v_action = 0.20
    elif abs_heading > 0.5:
        v_action = 0.40
    else:
        v_action = 0.80

    steer_action = torch.clamp(heading_error / 0.25, -1.0, 1.0)

    return torch.tensor(
        [[float(v_action), float(steer_action.item())]],
        dtype=torch.float32,
        device=obs_tensor.device,
    )


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavStaticEnvCfg()
    cfg.seed = args_cli.seed

    env = FWMiniNavStaticEnv(cfg=cfg)
    obs, _ = env.reset()

    print("[INFO] random-goal nav_static env reset done")
    print("[INFO] obs:", obs["policy"])
    print(f"[INFO] seed={args_cli.seed}")

    episode_count = 0
    success_count = 0
    timeout_count = 0
    failure_count = 0

    for i in range(args_cli.steps):
        # Warm up each episode after reset; use per-env episode length, not global step.
        episode_step = int(env.episode_length_buf[0].item())

        if episode_step < cfg.warmup_steps + 20:
            action = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=env.device)
        else:
            action = policy_from_obs(obs["policy"])

        obs, rew, terminated, truncated, info = env.step(action)

        obs_tensor = obs["policy"]
        goal_body_x = obs_tensor[0, 2].item()
        goal_body_y = obs_tensor[0, 3].item()
        goal_dist = obs_tensor[0, 4].item()
        heading_error = obs_tensor[0, 5].item()

        if i % args_cli.print_every == 0:
            root_pos = env.robot.data.root_pos_w[0, :3]
            root_vel = env.robot.data.root_lin_vel_w[0, :3]
            wheel_vel = env.robot.data.joint_vel[0, env.wheel_joint_ids]
            steer_pos = env.robot.data.joint_pos[0, env.steer_joint_ids]

            print(
                f"step {i} | "
                f"episode={episode_count} | "
                f"episode_step={episode_step} | "
                f"pos=({root_pos[0].item():.4f}, {root_pos[1].item():.4f}, {root_pos[2].item():.4f}) | "
                f"vel=({root_vel[0].item():.4f}, {root_vel[1].item():.4f}, {root_vel[2].item():.4f}) | "
                f"goal_body=({goal_body_x:.4f}, {goal_body_y:.4f}) | "
                f"dist={goal_dist:.4f} | heading_err={heading_error:.4f} | "
                f"wheel_vel={[round(v.item(), 4) for v in wheel_vel]} | "
                f"steer_pos={[round(v.item(), 4) for v in steer_pos]} | "
                f"action=({action[0, 0].item():.3f}, {action[0, 1].item():.3f}) | "
                f"rew={rew.item():.4f} | "
                f"terminated={terminated.item()} | truncated={truncated.item()} | "
                f"reason={env._last_done_reason}"
            )

        if terminated.item() or truncated.item():
            episode_count += 1

            reason = env._last_done_reason

            if reason == "goal":
                success_count += 1
            elif reason == "timeout":
                timeout_count += 1
            else:
                failure_count += 1

            success_rate = success_count / max(episode_count, 1)

            print(
                f"[EPISODE END] step={i} | "
                f"reason={reason} | "
                f"goal_dist={env._last_goal_dist:.4f} | "
                f"z_drop={env._last_z_drop:.5f} | "
                f"max_wheel={env._last_max_wheel:.4f} | "
                f"max_steer={env._last_max_steer:.4f} | "
                f"episodes={episode_count} | "
                f"success={success_count} | "
                f"timeout={timeout_count} | "
                f"failure={failure_count} | "
                f"success_rate={success_rate:.3f}"
            )

            obs, _ = env.reset()

    print("\n================ SUMMARY ================")
    print(f"episodes     : {episode_count}")
    print(f"success      : {success_count}")
    print(f"timeout      : {timeout_count}")
    print(f"failure      : {failure_count}")
    if episode_count > 0:
        print(f"success_rate : {success_count / episode_count:.3f}")
    print("=========================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
