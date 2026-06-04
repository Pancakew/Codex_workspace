from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class FWMiniNavObstacleVecEnvCfg(DirectRLEnvCfg):
    """Configuration for the vectorized FW-mini static-obstacle navigation task.

    This file does not run the simulation. It only stores the numbers used by
    `fwmini_nav_obstacle_vec_env.py`: physics step, robot geometry, action scale,
    goal range, obstacle geometry, safety thresholds, and reward weights.
    """

    # IsaacLab repeats each action for `decimation` physics steps.
    decimation = 2

    # Maximum episode duration. IsaacLab converts this to max_episode_length.
    episode_length_s = 55.0

    # Policy input/output sizes. Observation is built in `_get_observations`.
    action_space = 2
    observation_space = 21
    state_space = 0

    # Physics timestep and render frequency.
    sim = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=2,
    )

    # Number of parallel copies and their spacing in the world.
    scene = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=7.0,
    )

    # Regex path used when IsaacLab clones the robot into every env.
    robot_prim_path = "/World/envs/env_.*/Robot"

    # FW-mini chassis geometry and joint safety limits.
    max_speed_mps = 0.6
    max_steer_rad = 0.30
    wheel_radius = 0.07
    wheel_base = 0.35
    track_width = 0.25
    steer_rate_limit = 0.5

    # Policy action scaling:
    # action[0] in [0, 1] -> v_cmd in [0, action_v_scale]
    # action[1] in [-1, 1] -> steer_cmd in [-action_steer_scale, action_steer_scale]
    action_v_scale = 0.5
    action_steer_scale = 0.25

    # Reset/warmup values. Warmup keeps actions/rewards at zero while the robot settles.
    warmup_steps = 60
    settled_z_ref = -0.0100

    # Random goal range relative to each env origin.
    goal_x_min = 4.0
    goal_x_max = 5.2
    goal_y_min = -1.2
    goal_y_max = 1.2
    goal_radius = 0.35

    # Diagnostic option.
    # If True, all envs use the same fixed target.
    use_fixed_goal = False
    fixed_goal_x = 4.8
    fixed_goal_y = 0.8

    # Obstacles are `(center_x, center_y, size_x, size_y, size_z)` in local env coordinates.
    obstacles = (
        (2.10, 0.0, 0.30, 0.45, 0.40),
    )

    # Robot radius and obstacle clearance parameters used by `_obstacle_terms`.
    robot_radius = 0.22
    obstacle_collision_margin = 0.02
    obstacle_safe_distance = 0.65

    # Done/safety thresholds.
    collapse_z_drop = 0.018

    max_wheel_vel = 14.0
    max_steer_pos = 0.34

    # Reward weights used by `_get_rewards`.
    progress_reward_scale = 8.0
    goal_reward = 30.0
    collision_penalty = -30.0
    distance_penalty_scale = 0.04
    action_penalty_scale = 0.02
    steer_penalty_scale = 0.03
    obstacle_penalty_scale = 0.8
