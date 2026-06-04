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


parser = argparse.ArgumentParser(description="Run FW-mini obstacle nav smoke test.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--print-every", type=int, default=100)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)

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


def policy_from_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
    """
    Heuristic obstacle-aware policy for smoke test.

    This is not the final learned policy.
    It only verifies whether obstacle observations and collision logic are usable.
    """
    goal_dist = obs_tensor[:, 4]
    heading_error = obs_tensor[:, 5]

    obs_body_x = obs_tensor[:, 18]
    obs_body_y = obs_tensor[:, 19]
    obs_clearance = obs_tensor[:, 20]

    abs_heading = torch.abs(heading_error)

    v_action = torch.where(
        abs_heading > 1.0,
        torch.full_like(abs_heading, 0.20),
        torch.where(
            abs_heading > 0.5,
            torch.full_like(abs_heading, 0.40),
            torch.full_like(abs_heading, 0.75),
        ),
    )

    obstacle_ahead = (obs_body_x > 0.0) & (obs_body_x < 1.5) & (obs_clearance < 0.75)

    # If obstacle is on left, steer right; if obstacle is on right, steer left.
    avoid_dir = torch.where(obs_body_y >= 0.0, -1.0, 1.0)
    avoid_strength = torch.clamp((0.75 - obs_clearance) / 0.75, 0.0, 1.0)

    steer_goal = heading_error / 0.25
    steer_avoid = avoid_dir * avoid_strength * 1.2

    steer_action = torch.where(
        obstacle_ahead,
        steer_goal + steer_avoid,
        steer_goal,
    )

    steer_action = torch.clamp(steer_action, -1.0, 1.0)

    v_action = torch.where(
        obstacle_ahead,
        torch.minimum(v_action, torch.full_like(v_action, 0.35)),
        v_action,
    )
    v_action = torch.where(goal_dist < 0.35, torch.zeros_like(v_action), v_action)

    return torch.stack([v_action, steer_action], dim=1)


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavObstacleVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs

    env = FWMiniNavObstacleVecEnv(cfg=cfg)
    obs, _ = env.reset()

    print("[INFO] obstacle nav env reset done")
    print(f"[INFO] seed={args_cli.seed}, num_envs={args_cli.num_envs}")
    print("[INFO] obs shape:", obs["policy"].shape)

    total_episodes = 0
    success = 0
    collision = 0
    timeout = 0
    bad_z = 0
    bad_wheel = 0
    bad_steer = 0
    unknown_failure = 0

    for step in range(args_cli.steps):
        obs_tensor = obs["policy"]
        actions = policy_from_obs(obs_tensor)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        actions[warmup_mask, :] = 0.0

        obs, rew, terminated, truncated, info = env.step(actions)

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if step % args_cli.print_every == 0:
            obs0 = obs["policy"][0]
            root_pos = env.robot.data.root_pos_w[0, :3]
            root_vel = env.robot.data.root_lin_vel_w[0, :3]

            print(
                f"step {step} | "
                f"env0_pos=({root_pos[0].item():.4f}, {root_pos[1].item():.4f}, {root_pos[2].item():.4f}) | "
                f"env0_vel=({root_vel[0].item():.4f}, {root_vel[1].item():.4f}, {root_vel[2].item():.4f}) | "
                f"env0_dist={obs0[4].item():.4f} | "
                f"env0_heading={obs0[5].item():.4f} | "
                f"env0_obs=({obs0[18].item():.4f}, {obs0[19].item():.4f}, {obs0[20].item():.4f}) | "
                f"mean_dist={obs['policy'][:, 4].mean().item():.4f} | "
                f"done_this_step={len(done_ids)} | "
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

            for rid, reason_code in zip(done_ids.detach().cpu().tolist(), reasons):
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

                success_rate = success / max(total_episodes, 1)
                collision_rate = collision / max(total_episodes, 1)

                print(
                    f"[EPISODE END] global_step={step} | env_id={rid} | "
                    f"reason={reason} | "
                    f"goal_dist={env.last_goal_dist_buf[rid].item():.4f} | "
                    f"clearance={env.last_obstacle_clearance_buf[rid].item():.4f} | "
                    f"z_drop={env.last_z_drop_buf[rid].item():.5f} | "
                    f"max_wheel={env.last_max_wheel_buf[rid].item():.4f} | "
                    f"max_steer={env.last_max_steer_buf[rid].item():.4f} | "
                    f"episodes={total_episodes} | "
                    f"success_rate={success_rate:.3f} | "
                    f"collision_rate={collision_rate:.3f}"
                )

    failure = bad_z + bad_wheel + bad_steer + unknown_failure

    print("\n================ OBSTACLE SUMMARY ================")
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
    print("==================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()