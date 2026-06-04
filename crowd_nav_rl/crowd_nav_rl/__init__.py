import gymnasium as gym

print(">>> crowd_nav_rl __init__.py executing")

from crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env import FWMiniVerifyEnv
from crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env_cfg import FWMiniVerifyEnvCfg

print(">>> imported FWMiniVerifyEnv and FWMiniVerifyEnvCfg")

gym.register(
    id="FWMini-Verify-Direct-v0",
    entry_point="crowd_nav_rl.tasks.locomotion.verify.fwmini_verify_env:FWMiniVerifyEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": FWMiniVerifyEnvCfg(),
    },
)

print(">>> registered FWMini-Verify-Direct-v0")