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


parser = argparse.ArgumentParser(description="Run FW-mini verify env full smoke test.")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--print-every", type=int, default=50)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env_cfg import FWMiniVerifyEnvCfg
from crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env import FWMiniVerifyEnv


def main():
    cfg = FWMiniVerifyEnvCfg()
    env = FWMiniVerifyEnv(cfg=cfg)

    obs, _ = env.reset()

    print("[INFO] verify env reset done")
    print("[INFO] obs:", obs["policy"])

    for i in range(args_cli.steps):
        if i < 80:
            action = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=env.device)
        elif i < 230:
            action = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=env.device)
        elif i < 330:
            action = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=env.device)
        elif i < 420:
            action = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=env.device)
        elif i < 570:
            action = torch.tensor([[0.6, 1.0]], dtype=torch.float32, device=env.device)
        elif i < 670:
            action = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=env.device)
        elif i < 760:
            action = torch.tensor([[0.0, -1.0]], dtype=torch.float32, device=env.device)
        else:
            action = torch.tensor([[0.6, -1.0]], dtype=torch.float32, device=env.device)

        obs, rew, terminated, truncated, info = env.step(action)

        if i % args_cli.print_every == 0:
            root_pos = env.robot.data.root_pos_w[0, :3]
            root_vel = env.robot.data.root_lin_vel_w[0, :3]
            wheel_vel = env.robot.data.joint_vel[0, env.wheel_joint_ids]
            steer_pos = env.robot.data.joint_pos[0, env.steer_joint_ids]

            print(
                f"step {i} | "
                f"pos=({root_pos[0].item():.4f}, {root_pos[1].item():.4f}, {root_pos[2].item():.4f}) | "
                f"vel=({root_vel[0].item():.4f}, {root_vel[1].item():.4f}, {root_vel[2].item():.4f}) | "
                f"wheel_vel={[round(v.item(), 4) for v in wheel_vel]} | "
                f"steer_pos={[round(v.item(), 4) for v in steer_pos]} | "
                f"rew={rew.item():.4f} | "
                f"terminated={terminated.item()} | truncated={truncated.item()}"
            )

        if terminated.item() or truncated.item():
            print(
                f"[INFO] reset at step {i}, "
                f"terminated={terminated.item()}, truncated={truncated.item()}"
            )
            obs, _ = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()