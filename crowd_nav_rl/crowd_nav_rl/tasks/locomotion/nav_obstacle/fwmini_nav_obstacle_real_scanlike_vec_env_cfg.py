"""Config for the scan-like FW-mini obstacle navigation environment."""

from isaaclab.utils import configclass

from crowd_nav_rl.observations.fwmini_obs_contract import OBS_DIM_MAIN_SCAN44
from crowd_nav_rl.tasks.locomotion.nav_obstacle.fwmini_nav_obstacle_vec_env_cfg import (
    FWMiniNavObstacleVecEnvCfg,
)


@configclass
class FWMiniNavObstacleRealScanlikeVecEnvCfg(FWMiniNavObstacleVecEnvCfg):
    """Same dynamics/reward/done as obstacle env, but policy obs is `main_scan44`."""

    observation_space = OBS_DIM_MAIN_SCAN44
