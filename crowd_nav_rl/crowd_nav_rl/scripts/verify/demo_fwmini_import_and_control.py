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


parser = argparse.ArgumentParser(
    description="Demo: start Isaac Lab, import FW-mini, and run motion control."
)
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--print-every", type=int, default=60)
parser.add_argument("--hold-open", action="store_true", default=True)

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


def apply_cmd(robot, controller, wheel_ids, steer_ids, v_cmd, steer_cmd, dt, gear=6):
    steer_angles, wheel_omegas = controller.step(
        v_cmd=v_cmd,
        steer_cmd=steer_cmd,
        dt=dt,
        gear=gear,
    )

    steer_tensor = torch.tensor(
        [steer_angles],
        dtype=torch.float32,
        device=robot.device,
    )
    wheel_tensor = torch.tensor(
        [wheel_omegas],
        dtype=torch.float32,
        device=robot.device,
    )

    robot.set_joint_position_target(steer_tensor, joint_ids=steer_ids)
    robot.set_joint_velocity_target(wheel_tensor, joint_ids=wheel_ids)
    robot.write_data_to_sim()


def zero_cmd(robot, wheel_ids, steer_ids):
    zero_wheel = torch.zeros(
        (1, len(wheel_ids)),
        dtype=torch.float32,
        device=robot.device,
    )
    zero_steer = torch.zeros(
        (1, len(steer_ids)),
        dtype=torch.float32,
        device=robot.device,
    )

    robot.set_joint_velocity_target(zero_wheel, joint_ids=wheel_ids)
    robot.set_joint_position_target(zero_steer, joint_ids=steer_ids)
    robot.write_data_to_sim()


def print_state(robot, tag):
    pos = robot.data.root_pos_w[0, :3]
    vel = robot.data.root_lin_vel_w[0, :3]

    print(
        f"{tag} | "
        f"pos=({pos[0].item():.3f}, {pos[1].item():.3f}, {pos[2].item():.3f}) | "
        f"vel=({vel[0].item():.3f}, {vel[1].item():.3f}, {vel[2].item():.3f})"
    )


def run_segment(
    sim,
    robot,
    controller,
    wheel_ids,
    steer_ids,
    name,
    duration_sec,
    v_cmd,
    steer_cmd,
    dt,
    gear=6,
):
    steps = int(duration_sec / dt)

    print(f"\n=== {name} ===")
    print(
        f"name={name}, duration={duration_sec:.2f}s, "
        f"v_cmd={v_cmd:.3f}, steer_cmd={steer_cmd:.3f}, gear={gear}"
    )

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

        if i % args_cli.print_every == 0:
            print_state(robot, f"{name} step {i}")


def main():
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=4,
    )
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

    print("\n[INFO] Isaac Lab simulation started.")
    print("[INFO] FW-mini loaded at: /World/fw_mini")

    wheel_ids, wheel_names = robot.find_joints(".*_wheel_joint")
    steer_ids, steer_names = robot.find_joints(".*_steering_hinge_joint")

    print("\n[INFO] wheel joints:")
    for name in wheel_names:
        print("  -", name)

    print("\n[INFO] steering joints:")
    for name in steer_names:
        print("  -", name)

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
        raise RuntimeError(
            f"Wheel joint order mismatch. Actual: {list(wheel_names)}"
        )

    if list(steer_names) != expected_steers:
        raise RuntimeError(
            f"Steer joint order mismatch. Actual: {list(steer_names)}"
        )

    controller = FWMiniCmdVelController(
        FWMiniCmdVelConfig(
            wheel_base=0.35,
            track_width=0.25,
            wheel_radius=0.07,
            max_speed_mps=0.6,
            max_steer_rad=0.30,
            steer_rate_limit=0.5,
        )
    )
    controller.reset()

    print("\n[INFO] controller initialized.")
    print("[INFO] settle robot...")

    for _ in range(args_cli.settle_steps):
        zero_cmd(robot, wheel_ids, steer_ids)
        sim.step()
        robot.update(dt)

    print_state(robot, "after settle")

    motion_sequence = [
        {
            "name": "forward_drive",
            "duration": 2.5,
            "v_cmd": 0.45,
            "steer_cmd": 0.0,
            "gear": 6,
        },
        {
            "name": "stop_1",
            "duration": 1.0,
            "v_cmd": 0.0,
            "steer_cmd": 0.0,
            "gear": 6,
        },
        {
            "name": "left_align",
            "duration": 1.0,
            "v_cmd": 0.0,
            "steer_cmd": 0.25,
            "gear": 6,
        },
        {
            "name": "left_drive",
            "duration": 2.5,
            "v_cmd": 0.35,
            "steer_cmd": 0.25,
            "gear": 6,
        },
        {
            "name": "left_reset",
            "duration": 1.0,
            "v_cmd": 0.0,
            "steer_cmd": 0.0,
            "gear": 6,
        },
        {
            "name": "right_align",
            "duration": 1.0,
            "v_cmd": 0.0,
            "steer_cmd": -0.25,
            "gear": 6,
        },
        {
            "name": "right_drive",
            "duration": 2.5,
            "v_cmd": 0.35,
            "steer_cmd": -0.25,
            "gear": 6,
        },
        {
            "name": "final_stop",
            "duration": 1.5,
            "v_cmd": 0.0,
            "steer_cmd": 0.0,
            "gear": 6,
        },
    ]

    print("\n[INFO] start motion sequence.")

    for seg in motion_sequence:
        run_segment(
            sim=sim,
            robot=robot,
            controller=controller,
            wheel_ids=wheel_ids,
            steer_ids=steer_ids,
            name=seg["name"],
            duration_sec=seg["duration"],
            v_cmd=seg["v_cmd"],
            steer_cmd=seg["steer_cmd"],
            dt=dt,
            gear=seg["gear"],
        )

    zero_cmd(robot, wheel_ids, steer_ids)
    print("\n[INFO] motion sequence finished.")
    print("[INFO] The window will stay open. Close Isaac Sim window to exit.")

    while simulation_app.is_running():
        zero_cmd(robot, wheel_ids, steer_ids)
        sim.step()
        robot.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
