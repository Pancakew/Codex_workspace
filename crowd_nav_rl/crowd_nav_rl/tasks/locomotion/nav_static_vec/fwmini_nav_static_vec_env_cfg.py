from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class FWMiniNavStaticVecEnvCfg(DirectRLEnvCfg):
    """FW-mini vectorized static random-goal navigation environment."""

    decimation = 2
    episode_length_s = 40.0

    action_space = 2
    observation_space = 18
    state_space = 0

    sim = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=2,
    )

    scene = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=6.0,
    )

    robot_prim_path = "/World/envs/env_.*/Robot"

    # controller
    max_speed_mps = 0.6
    max_steer_rad = 0.30
    wheel_radius = 0.07
    wheel_base = 0.35
    track_width = 0.25
    steer_rate_limit = 0.5

    # action scaling
    action_v_scale = 0.5
    action_steer_scale = 0.25

    # reset / warmup
    warmup_steps = 60
    settled_z_ref = -0.0100

    # random goal range, relative to each env origin
    goal_x_min = 2.5
    goal_x_max = 5.0
    goal_y_min = -1.5
    goal_y_max = 1.5
    goal_radius = 0.35

    # safety
    collapse_z_drop = 0.018
    max_wheel_vel = 10.0
    max_steer_pos = 0.34

    # reward
    progress_reward_scale = 8.0
    goal_reward = 25.0
    distance_penalty_scale = 0.04
    action_penalty_scale = 0.02
    steer_penalty_scale = 0.03