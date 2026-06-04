import argparse
import sys
from pathlib import Path
import torch


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="FW-mini three-stage cmd_vel verification.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--forward-steps", type=int, default=180)
parser.add_argument("--align-steps", type=int, default=90)
parser.add_argument("--turn-steps", type=int, default=180)
parser.add_argument("--hold-stop-steps", type=int, default=60)
parser.add_argument("--reset-steps", type=int, default=90)
parser.add_argument("--print-every", type=int, default=20)

parser.add_argument("--forward-v", type=float, default=0.5)
parser.add_argument("--turn-v", type=float, default=0.3)
parser.add_argument("--turn-steer", type=float, default=0.25)

parser.add_argument("--collapse-z-drop", type=float, default=0.018)
parser.add_argument("--max-wheel-vel", type=float, default=10.0)
parser.add_argument("--max-steer-pos", type=float, default=0.34)
parser.add_argument("--max-align-drift", type=float, default=0.05)
parser.add_argument("--max-align-planar-speed", type=float, default=0.08)
parser.add_argument("--stuck-window", type=int, default=20)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation

from crowd_nav_rl.assets.fw_mini.config.fw_mini_cfg import FW_MINI_CFG
from crowd_nav_rl.controllers.fwmini_cmdvel_controller import (
    FWMiniCmdVelController,
    FWMiniCmdVelConfig,
)


def get_state(robot, wheel_ids, steer_ids, dt):
    robot.update(dt)
    pos = robot.data.root_pos_w[0, :3].clone()
    vel = robot.data.root_lin_vel_w[0, :3].clone()
    wheel_vel = robot.data.joint_vel[0, wheel_ids].reshape(-1).clone()
    steer_pos = robot.data.joint_pos[0, steer_ids].reshape(-1).clone()
    return pos, vel, wheel_vel, steer_pos


def print_state(robot, wheel_ids, steer_ids, tag, dt):
    pos, vel, wheel_vel, steer_pos = get_state(robot, wheel_ids, steer_ids, dt)

    print(
        f"{tag} | "
        f"pos=({pos[0].item():.4f}, {pos[1].item():.4f}, {pos[2].item():.4f}) | "
        f"vel=({vel[0].item():.4f}, {vel[1].item():.4f}, {vel[2].item():.4f}) | "
        f"wheel_vel={[round(v.item(), 4) for v in wheel_vel]} | "
        f"steer_pos={[round(v.item(), 4) for v in steer_pos]}"
    )


def apply_cmd(robot, controller, wheel_ids, steer_ids, v_cmd, steer_cmd, dt, gear=6):
    steer_angles, wheel_omegas = controller.step(v_cmd, steer_cmd, dt, gear=gear)

    steer_tensor = torch.tensor([steer_angles], dtype=torch.float32, device=robot.device)
    wheel_tensor = torch.tensor([wheel_omegas], dtype=torch.float32, device=robot.device)

    robot.set_joint_position_target(steer_tensor, joint_ids=steer_ids)
    robot.set_joint_velocity_target(wheel_tensor, joint_ids=wheel_ids)
    robot.write_data_to_sim()

    return steer_angles, wheel_omegas


def zero_cmd(robot, wheel_ids, steer_ids):
    zero_wheel = torch.zeros((1, len(wheel_ids)), dtype=torch.float32, device=robot.device)
    zero_steer = torch.zeros((1, len(steer_ids)), dtype=torch.float32, device=robot.device)

    robot.set_joint_velocity_target(zero_wheel, joint_ids=wheel_ids)
    robot.set_joint_position_target(zero_steer, joint_ids=steer_ids)
    robot.write_data_to_sim()


def run_segment(
    sim,
    robot,
    controller,
    wheel_ids,
    steer_ids,
    name,
    v_cmd,
    steer_cmd,
    steps,
    dt,
    settled_z,
    gear=6,
    check_align=False,
):
    print(f"\n=== {name} ===")

    stuck_count = 0

    max_abs_z_drop = 0.0
    max_abs_wheel_vel = 0.0
    max_abs_steer_pos = 0.0
    max_planar_speed = 0.0
    max_align_drift = 0.0

    start_pos, _, _, _ = get_state(robot, wheel_ids, steer_ids, dt)
    start_x = float(start_pos[0].item())
    start_y = float(start_pos[1].item())

    for i in range(steps):
        apply_cmd(
            robot=robot,
            controller=controller,
            wheel_ids=wheel_ids,
            steer_ids=steer_ids,
            v_cmd=v_cmd,
            steer_cmd=steer_cmd,
            dt=dt,
            gear=gear,
        )

        sim.step()
        robot.update(dt)

        pos, vel, wheel_vel, steer_pos = get_state(robot, wheel_ids, steer_ids, dt)

        x_now = float(pos[0].item())
        y_now = float(pos[1].item())
        z_now = float(pos[2].item())

        z_drop = settled_z - z_now
        planar_speed = abs(float(vel[0].item())) + abs(float(vel[1].item()))
        max_wheel = float(torch.max(torch.abs(wheel_vel)).item())
        max_steer = float(torch.max(torch.abs(steer_pos)).item())
        align_drift = ((x_now - start_x) ** 2 + (y_now - start_y) ** 2) ** 0.5

        max_abs_z_drop = max(max_abs_z_drop, abs(z_drop))
        max_abs_wheel_vel = max(max_abs_wheel_vel, max_wheel)
        max_abs_steer_pos = max(max_abs_steer_pos, max_steer)
        max_planar_speed = max(max_planar_speed, planar_speed)
        max_align_drift = max(max_align_drift, align_drift)

        if abs(z_drop) > args_cli.collapse_z_drop:
            print_state(robot, wheel_ids, steer_ids, f"{name} step {i}", dt)
            print(
                f"[FAIL] {name}: z drop too large. "
                f"settled_z={settled_z:.4f}, z={z_now:.4f}, drop={z_drop:.4f}"
            )
            return False

        if (not check_align) and max_wheel > args_cli.max_wheel_vel:
            print_state(robot, wheel_ids, steer_ids, f"{name} step {i}", dt)
            print(
                f"[FAIL] {name}: wheel velocity spike. "
                f"max_wheel={max_wheel:.4f} > {args_cli.max_wheel_vel:.4f}"
            )
            return False

        if max_steer > args_cli.max_steer_pos:
            print_state(robot, wheel_ids, steer_ids, f"{name} step {i}", dt)
            print(
                f"[FAIL] {name}: steering overshoot. "
                f"max_steer={max_steer:.4f} > {args_cli.max_steer_pos:.4f}"
            )
            return False

        if check_align:
            if align_drift > args_cli.max_align_drift:
                print_state(robot, wheel_ids, steer_ids, f"{name} step {i}", dt)
                print(
                    f"[FAIL] {name}: align drift too large. "
                    f"drift={align_drift:.4f} > {args_cli.max_align_drift:.4f}"
                )
                return False

            if i > 10 and planar_speed > args_cli.max_align_planar_speed:
                print_state(robot, wheel_ids, steer_ids, f"{name} step {i}", dt)
                print(
                    f"[FAIL] {name}: align planar speed too large. "
                    f"planar_speed={planar_speed:.4f} > {args_cli.max_align_planar_speed:.4f}"
                )
                return False

        if (not check_align) and abs(v_cmd) > 1e-4 and max_wheel > 1.0 and planar_speed < 0.01:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= args_cli.stuck_window:
            print_state(robot, wheel_ids, steer_ids, f"{name} step {i}", dt)
            print(f"[FAIL] {name}: wheels spinning but base is not moving.")
            return False

        if i % args_cli.print_every == 0:
            print_state(robot, wheel_ids, steer_ids, f"{name} step {i}", dt)

    print(
        f"[SUMMARY] {name}: "
        f"max_z_drop={max_abs_z_drop:.5f}, "
        f"max_wheel_vel={max_abs_wheel_vel:.4f}, "
        f"max_steer_pos={max_abs_steer_pos:.4f}, "
        f"max_planar_speed={max_planar_speed:.4f}, "
        f"max_align_drift={max_align_drift:.4f}"
    )

    return True


def main():
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    sim = SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.6,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        )
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/DomeLight", light_cfg)

    robot_cfg = FW_MINI_CFG.replace(prim_path="/World/fw_mini")
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    dt = sim.get_physics_dt()
    robot.update(dt)

    wheel_ids, wheel_names = robot.find_joints(".*_wheel_joint")
    steer_ids, steer_names = robot.find_joints(".*_steering_hinge_joint")

    print("wheel joints:", wheel_names)
    print("steer joints:", steer_names)

    expected_wheels = [
        "left_front_wheel_joint",
        "left_rear_wheel_joint",
        "right_rear_wheel_joint",
        "right_front_wheel_joint",
    ]
    expected_steers = [
        "left_steering_hinge_joint",
        "rear_left_steering_hinge_joint",
        "rear_right_steering_hinge_joint",
        "right_steering_hinge_joint",
    ]

    if list(wheel_names) != expected_wheels:
        raise RuntimeError(f"Wheel joint order mismatch. Actual: {list(wheel_names)}")

    if list(steer_names) != expected_steers:
        raise RuntimeError(f"Steer joint order mismatch. Actual: {list(steer_names)}")

    controller = FWMiniCmdVelController(
        FWMiniCmdVelConfig(
            max_speed_mps=0.6,
            max_steer_rad=0.30,
            steer_rate_limit=0.5,
            wheel_radius=0.07,
        )
    )
    controller.reset()

    print("\n=== settle ===")
    zero_cmd(robot, wheel_ids, steer_ids)

    for _ in range(args_cli.settle_steps):
        zero_cmd(robot, wheel_ids, steer_ids)
        sim.step()
        robot.update(dt)

    print_state(robot, wheel_ids, steer_ids, "initial", dt)

    pos, _, _, _ = get_state(robot, wheel_ids, steer_ids, dt)
    settled_z = float(pos[2].item())

    print(f"[INFO] settled_z = {settled_z:.4f}")
    print("[INFO] start three-stage verification")

    sequence = [
        ("forward_drive", args_cli.forward_v, 0.0, args_cli.forward_steps, False),
        ("forward_stop", 0.0, 0.0, args_cli.hold_stop_steps, True),
        ("left_align", 0.0, args_cli.turn_steer, args_cli.align_steps, True),
        ("left_drive", args_cli.turn_v, args_cli.turn_steer, args_cli.turn_steps, False),
        ("left_hold_stop", 0.0, args_cli.turn_steer, args_cli.hold_stop_steps, True),
        ("left_reset_steer", 0.0, 0.0, args_cli.reset_steps, True),
        ("right_align", 0.0, -args_cli.turn_steer, args_cli.align_steps, True),
        ("right_drive", args_cli.turn_v, -args_cli.turn_steer, args_cli.turn_steps, False),
        ("right_hold_stop", 0.0, -args_cli.turn_steer, args_cli.hold_stop_steps, True),
        ("right_reset_steer", 0.0, 0.0, args_cli.reset_steps, True),
    ]

    ok = True
    for name, v_cmd, steer_cmd, steps, check_align in sequence:
        ok = run_segment(
            sim=sim,
            robot=robot,
            controller=controller,
            wheel_ids=wheel_ids,
            steer_ids=steer_ids,
            name=name,
            v_cmd=v_cmd,
            steer_cmd=steer_cmd,
            steps=steps,
            dt=dt,
            settled_z=settled_z,
            gear=6,
            check_align=check_align,
        )
        if not ok:
            break

    print("\n====================================================")
    if ok:
        print("[PASS] FW-mini three-stage cmd_vel verification finished.")
    else:
        print("[FAIL] FW-mini three-stage cmd_vel verification failed.")
    print("====================================================\n")

    simulation_app.close()


if __name__ == "__main__":
    main()