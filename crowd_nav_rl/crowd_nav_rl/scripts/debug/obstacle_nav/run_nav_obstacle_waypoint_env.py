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


parser = argparse.ArgumentParser(description="Run FW-mini obstacle waypoint smoke test.")
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


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def body_target_from_world(env, target_w: torch.Tensor):
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
    heading = torch.atan2(body_y, body_x)
    heading = wrap_to_pi(heading)

    return body_x, body_y, dist, heading


class WaypointState:
    """Per-environment waypoint state.

    phase:
        0 -> go to side waypoint
        1 -> go to forward waypoint
        2 -> go to final goal
    """

    def __init__(self, num_envs: int, device):
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pass_side = torch.ones(num_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor):
        self.phase[env_ids] = 0
        self.pass_side[env_ids] = 1.0


def waypoint_policy(env, obs_tensor: torch.Tensor, wp_state: WaypointState) -> torch.Tensor:
    device = obs_tensor.device
    num_envs = obs_tensor.shape[0]

    root_pos = env.robot.data.root_pos_w[:, :3]
    origins = env.scene.env_origins

    goal_body_y = obs_tensor[:, 3]
    goal_dist = obs_tensor[:, 4]
    heading_error_to_goal = obs_tensor[:, 5]

    local_x = root_pos[:, 0] - origins[:, 0]
    local_y = root_pos[:, 1] - origins[:, 1]

    obs_body_x = obs_tensor[:, 18]
    obs_clearance = obs_tensor[:, 20]

    main_obs = env.cfg.obstacles[0]
    obs_cx, obs_cy, obs_sx, obs_sy, _ = main_obs

    # Set pass side at beginning. If goal is above, pass above; otherwise pass below.
    new_side = torch.where(goal_body_y >= 0.0, 1.0, -1.0)
    wp_state.pass_side = torch.where(wp_state.phase == 0, new_side, wp_state.pass_side)

    # Three-stage waypoint plan.
    # p0: side waypoint before/around obstacle
    # p1: forward waypoint after obstacle
    # p2: final goal
    side_y = wp_state.pass_side * 1.05

    p0 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    p0[:, 0] = origins[:, 0] + obs_cx + 0.35
    p0[:, 1] = origins[:, 1] + side_y

    p1 = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    p1[:, 0] = origins[:, 0] + obs_cx + 1.25
    p1[:, 1] = origins[:, 1] + side_y

    # Update phase based on local position.
    # Once close to side lane and near obstacle x, move to phase 1.
    reached_side_lane = torch.abs(local_y - side_y) < 0.22
    passed_near_obstacle = local_x > (obs_cx + 0.20)

    to_phase1 = (wp_state.phase == 0) & reached_side_lane & passed_near_obstacle
    wp_state.phase = torch.where(to_phase1, torch.ones_like(wp_state.phase), wp_state.phase)

    # Once past obstacle enough, go to final goal.
    passed_obstacle = local_x > (obs_cx + 1.10)
    to_phase2 = (wp_state.phase == 1) & passed_obstacle
    wp_state.phase = torch.where(to_phase2, torch.ones_like(wp_state.phase) * 2, wp_state.phase)

    # Select target.
    target_w = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)

    target_w = torch.where((wp_state.phase == 0).unsqueeze(1), p0, target_w)
    target_w = torch.where((wp_state.phase == 1).unsqueeze(1), p1, target_w)

    # For phase 2, use actual goal from env.
    goal_w = env._goal_w
    target_w = torch.where((wp_state.phase == 2).unsqueeze(1), goal_w, target_w)

    _, _, target_dist, target_heading = body_target_from_world(env, target_w)

    # Control law.
    abs_heading = torch.abs(target_heading)

    v_action = torch.where(
        abs_heading > 1.0,
        torch.full_like(abs_heading, 0.12),
        torch.where(
            abs_heading > 0.55,
            torch.full_like(abs_heading, 0.25),
            torch.full_like(abs_heading, 0.55),
        ),
    )

    # Slow down near obstacle.
    near_obstacle = (obs_body_x > -0.2) & (obs_body_x < 1.2) & (obs_clearance < 0.60)
    v_action = torch.where(
        near_obstacle,
        torch.minimum(v_action, torch.full_like(v_action, 0.20)),
        v_action,
    )

    v_action = torch.where(goal_dist < env.cfg.goal_radius, torch.zeros_like(v_action), v_action)

    steer_action = torch.clamp(target_heading / 0.25, -1.0, 1.0)

    return torch.stack([v_action, steer_action], dim=1)


def main():
    set_seed(args_cli.seed)

    cfg = FWMiniNavObstacleVecEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs

    print("[INFO] cfg.obstacles:", cfg.obstacles)
    print("[INFO] cfg.robot_radius:", cfg.robot_radius)
    print("[INFO] cfg.obstacle_collision_margin:", cfg.obstacle_collision_margin)
    print("[INFO] cfg.goal range:", cfg.goal_x_min, cfg.goal_x_max, cfg.goal_y_min, cfg.goal_y_max)

    env = FWMiniNavObstacleVecEnv(cfg=cfg)
    obs, _ = env.reset()

    wp_state = WaypointState(num_envs=args_cli.num_envs, device=env.device)

    print("[INFO] obstacle waypoint smoke test reset done")
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
        actions = waypoint_policy(env, obs_tensor, wp_state)

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
                f"phase0={int(wp_state.phase[0].item())} | "
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

            wp_state.reset(done_ids)

    failure = bad_z + bad_wheel + bad_steer + unknown_failure

    print("\n================ OBSTACLE WAYPOINT SUMMARY ================")
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
    print("===========================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()