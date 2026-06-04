from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class FWMiniVerifyEnvCfg(DirectRLEnvCfg):
    """FW-mini minimal verify environment."""

    decimation = 2
    episode_length_s = 40.0

    action_space = 2
    observation_space = 16
    state_space = 0

    sim = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=2,
    )

    scene = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    robot_prim_path = "/World/envs/env_0/Robot"

    max_speed_mps = 0.6
    max_steer_rad = 0.30
    wheel_radius = 0.07
    wheel_base = 0.35
    track_width = 0.25
    steer_rate_limit = 0.5

    action_v_scale = 0.5
    action_steer_scale = 0.25

    warmup_steps = 60
    settled_z_ref = -0.0100

    collapse_z_drop = 0.018
    max_wheel_vel = 10.0
    max_steer_pos = 0.34