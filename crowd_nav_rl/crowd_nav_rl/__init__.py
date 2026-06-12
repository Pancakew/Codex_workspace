"""crowd_nav_rl package.

Keep package import lightweight so pure modules such as
`crowd_nav_rl.observations` can be reused by future ROS code without importing
IsaacLab. Isaac/Gym registration is optional and only runs when dependencies are
available.
"""

try:
    import gymnasium as gym

    from crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env import FWMiniVerifyEnv
    from crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env_cfg import FWMiniVerifyEnvCfg

    if "FWMini-Verify-Direct-v0" not in gym.envs.registry:
        gym.register(
            id="FWMini-Verify-Direct-v0",
            entry_point="crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env:FWMiniVerifyEnv",
            disable_env_checker=True,
            kwargs={
                "cfg": FWMiniVerifyEnvCfg(),
            },
        )
except ImportError:
    # IsaacLab is not required for pure observation-contract imports.
    pas