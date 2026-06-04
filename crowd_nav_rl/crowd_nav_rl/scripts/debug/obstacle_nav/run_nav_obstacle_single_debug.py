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


parser = argparse.ArgumentParser(description="Single-env obstacle waypoint debug.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--print-every", type=int, default=20)
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


class SingleWaypointState:
    def __init__(self, device):
        self.phase = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self):
        self.phase[:] = 0


def fixed_upper_waypoint_policy(env, obs_tensor, state):
    device = obs_tensor.device

    origins = env.scene.env_origins
    root_pos = env.robot.data.root_pos_w[:, :3]

    goal_dist = obs_tensor[:, 4]

    local_x = root_pos[:, 0] - origins[:, 0]
    local_y = root_pos[:, 1] - origins[:, 1]

    obs_cx, obs_cy, obs_sx, obs_sy, _ = env.cfg.obstacles[0]

    # Fixed upper-lane route:
    # phase 0 -> go to the upper waypoint near the obstacle.
    # phase 1 -> cross past the obstacle on that lane.
    # phase 2 -> drive to the final goal.
    p0 = torch.tensor([[obs_cx + 0.15, 1.05]], dtype=torch.float32, device=device)
    p1 = torch.tensor([[obs_cx + 1.35, 1.05]], dtype=torch.float32, device=device)
    p2 = env._goal_w.clone()

    p0[:, 0] += origins[:, 0]
    p0[:, 1] += origins[:, 1]

    p1[:, 0] += origins[:, 0]
    p1[:, 1] += origins[:, 1]

    # Phase switching logic for the fixed upper-lane route.
    if state.phase[0].item() == 0:
        if local_x[0].item() > obs_cx - 0.05 and local_y[0].item() > 0.75:
            state.phase[0] = 1

    if state.phase[0].item() == 1:
        if local_x[0].item() > obs_cx + 1.10:
            state.phase[0] = 2

    if state.phase[0].item() == 0:
        target = p0
    elif state.phase[0].item() == 1:
        target = p1
    else:
        target = p2

    _, _, target_dist, heading = body_target_from_world(env, target)

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

    action = torch.stack([v_action, steer_action], dim=1)

    return action, target


def extract_policy_obs(obs):
    """Return the policy observation tensor from DirectRLEnv output."""
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavObstacleVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = 1

    # Use a fixed goal to make this single-env diagnostic repeatable.
    cfg.use_fixed_goal = True
    cfg.fixed_goal_x = 4.8
    cfg.fixed_goal_y = 0.8

    print("[INFO] cfg.obstacles:", cfg.obstacles)
    print("[INFO] fixed goal:", cfg.fixed_goal_x, cfg.fixed_goal_y)
    print("[INFO] robot_radius:", cfg.robot_radius)
    print("[INFO] margin:", cfg.obstacle_collision_margin)

    env = FWMiniNavObstacleVecEnv(cfg=cfg)

    obs, _ = env.reset()
    obs_tensor = extract_policy_obs(obs)

    state = SingleWaypointState(device=env.device)

    print("[INFO] single obstacle debug reset done")
    print("[INFO] obs shape:", obs_tensor.shape)

    for step in range(args_cli.steps):
        action, target = fixed_upper_waypoint_policy(env, obs_tensor, state)

        warmup_mask = env.episode_length_buf < (cfg.warmup_steps + 20)
        action[warmup_mask, :] = 0.0

        obs, rew, terminated, truncated, info = env.step(action)
        obs_tensor = extract_policy_obs(obs)

        root_pos = env.robot.data.root_pos_w[0, :3]
        root_vel = env.robot.data.root_lin_vel_w[0, :3]

        wheel_vel = env.robot.data.joint_vel[0, env.wheel_joint_ids]
        steer_pos = env.robot.data.joint_pos[0, env.steer_joint_ids]

        o = obs_tensor[0]

        if step % args_cli.print_every == 0:
            print(
                f"step={step:04d} | "
                f"phase={int(state.phase[0].item())} | "
                f"pos=({root_pos[0].item():.3f},{root_pos[1].item():.3f},{root_pos[2].item():.3f}) | "
                f"vel=({root_vel[0].item():.3f},{root_vel[1].item():.3f},{root_vel[2].item():.3f}) | "
                f"goal_dist={o[4].item():.3f} | "
                f"goal_heading={o[5].item():.3f} | "
                f"obs_body=({o[18].item():.3f},{o[19].item():.3f},{o[20].item():.3f}) | "
                f"action=({action[0, 0].item():.3f},{action[0, 1].item():.3f}) | "
                f"target=({target[0, 0].item():.3f},{target[0, 1].item():.3f}) | "
                f"max_wheel={torch.max(torch.abs(wheel_vel)).item():.3f} | "
                f"max_steer={torch.max(torch.abs(steer_pos)).item():.3f}"
            )

        if terminated.item() or truncated.item():
            reason = REASON_MAP.get(int(env.done_reason_buf[0].item()), "unknown")

            print("\n[EPISODE END]")
            print(f"step={step}")
            print(f"reason={reason}")
            print(f"goal_dist={env.last_goal_dist_buf[0].item():.4f}")
            print(f"clearance={env.last_obstacle_clearance_buf[0].item():.4f}")
            print(f"z_drop={env.last_z_drop_buf[0].item():.5f}")
            print(f"max_wheel={env.last_max_wheel_buf[0].item():.4f}")
            print(f"max_steer={env.last_max_steer_buf[0].item():.4f}")
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
