"""IsaacLab asset configuration for the FW-mini robot.

This file tells IsaacLab where the FW-mini USD is and how its joints should be
actuated. It is not a navigation environment and does not contain reward or RL
logic. Environments import `FW_MINI_CFG` and place it into the scene.
"""

import isaaclab.sim as sim_utils
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


FW_MINI_USD_PATH = str(Path(__file__).resolve().parents[1] / "usd" / "fw_mini" / "fw_mini.usd")


FW_MINI_CFG = ArticulationCfg(
    # Load the converted USD robot and configure global rigid-body properties.
    spawn=sim_utils.UsdFileCfg(
        usd_path=FW_MINI_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.20),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),

    actuators={
        # Wheel joints are controlled by velocity targets.
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=2000.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=50.0,
        ),
        # Steering hinge joints are controlled by position targets.
        "steering_drive": ImplicitActuatorCfg(
            joint_names_expr=[".*_steering_hinge_joint"],
            effort_limit_sim=500.0,
            velocity_limit_sim=10.0,
            stiffness=3000.0,
            damping=300.0,
        ),
    },
)
