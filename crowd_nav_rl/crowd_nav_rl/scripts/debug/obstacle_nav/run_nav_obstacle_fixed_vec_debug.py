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


parser = argparse.ArgumentParser(description="Vectorized fixed-goal obstacle waypoint debug.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--print-every", type=int, default=100)
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_from_quat_wxyz(q):
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
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


class VecWaypointState:
    def __init__(self, num_envs, device):
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids):
        self.phase[env_ids] = 0


def fixed_upper_waypoint_policy(env, obs_tensor, state):
    device = obs_tensor.device
    num_envs = obs_tensor.shape[0]

    origins = env.scene.env_origins
    root_pos = env.robot.data.root_pos_w[:, :3]

    goal_dist = obs_tensor[:, 4]

    local_x = root_pos[:, 0] - origins[:, 0]
    local_y = root_pos[:, 1] - origins[:, 1]

    obs_cx, obs_cy, obs_sx, obs_sy, _ = env.cfg.obstacles[0]

    p0 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    p1 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)

    p0[:, 0] = origins[:, 0] + obs_cx + 0.15
    p0[:, 1] = origins[:, 1] + 1.05

    p1[:, 0] = origins[:, 0] + obs_cx + 1.35
    p1[:, 1] = origins[:, 1] + 1.05

    p2 = env._goal_w.clone()

    # phase 0 -> phase 1
    to_phase1 = (
        (state.phase == 0)
        & (local_x > obs_cx - 0.05)
        & (local_y > 0.75)
    )
    state.phase = torch.where(to_phase1, torch.ones_like(state.phase), state.phase)

    # phase 1 -> phase 2
    to_phase2 = (
        (state.phase == 1)
        & (local_x > obs_cx + 1.10)
    )
    state.phase = torch.where(to_phase2, torch.ones_like(state.phase) * 2, state.phase)

    target = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    target = torch.where((state.phase == 0).unsqueeze(1), p0, target)
    target = torch.where((state.phase == 1).unsqueeze(1), p1, target)
    target = torch.where((state.phase == 2).unsqueeze(1), p2, target)

    _, _, _, heading = body_target_from_world(env, target)

    abs_heading = torch.abs(heading)

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
    steer_action = torch.clamp(heading / 0.25, -1.0, 1.0)

    return torch.stack([v_action, steer_action], dim=1), target


def extract_policy_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavObstacleVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs

    cfg.use_fixed_goal = True
    cfg.fixed_goal_x = 4.8
    cfg.fixed_goal_y = 0.8

    print("[INFO] cfg.obstacles:", cfg.obstacles)
    print("[INFO] fixed goal:", cfg.fixed_goal_x, cfg.fixed_goal_y)
    print("[INFO] robot_radius:", cfg.robot_radius)
    print("[INFO] margin:", cfg.obstacle_collision_margin)
    print("[INFO] num_envs:", cfg.scene.num_envs)

    env = FWMiniNavObstacleVecEnv(cfg=cfg)
    obs, _ = env.reset()
    obs_tensor = extract_policy_obs(obs)

    state = VecWaypointState(num_envs=args_cli.num_envs, device=env.device)

    total_episodes = 0
    success = 0
    collision = 0
    timeout = 0
    bad_z = 0
    bad_wheel = 0
    bad_steer = 0
    unknown_failure = 0

    print("[INFO] fixed vec obstacle debug reset done")
    print("[INFO] obs shape:", obs_tensor.shape)

    for step in range(args_cli.steps):
        action, target = fixed_upper_waypoint_policy(env, obs_tensor, state)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        action[warmup_mask, :] = 0.0

        obs, rew, terminated, truncated, info = env.step(action)
        obs_tensor = extract_policy_obs(obs)

        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)

        if step % args_cli.print_every == 0:
            o0 = obs_tensor[0]
            root_pos0 = env.robot.data.root_pos_w[0, :3]
            root_vel0 = env.robot.data.root_lin_vel_w[0, :3]

            print(
                f"step={step:04d} | "
                f"env0_phase={int(state.phase[0].item())} | "
                f"env0_pos=({root_pos0[0].item():.3f},{root_pos0[1].item():.3f},{root_pos0[2].item():.3f}) | "
                f"env0_vel=({root_vel0[0].item():.3f},{root_vel0[1].item():.3f},{root_vel0[2].item():.3f}) | "
                f"env0_goal_dist={o0[4].item():.3f} | "
                f"env0_obs=({o0[18].item():.3f},{o0[19].item():.3f},{o0[20].item():.3f}) | "
                f"phase_count=[{int((state.phase == 0).sum().item())}, "
                f"{int((state.phase == 1).sum().item())}, "
                f"{int((state.phase == 2).sum().item())}] | "
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

            state.reset(done_ids)

    failure = bad_z + bad_wheel + bad_steer + unknown_failure

    print("\n================ FIXED VEC OBSTACLE SUMMARY ================")
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
    print("============================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()