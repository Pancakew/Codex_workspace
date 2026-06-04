fw-mini.urdf / xacro：
    小车原始机器人描述。

meshes：
    小车视觉/碰撞网格。

fw_mini.usd：
    Isaac Sim / Isaac Lab 实际加载的资产。

fw_mini_cfg.py：
    Isaac Lab 的 ArticulationCfg。
    它告诉 Isaac Lab：
        小车 USD 在哪里；
        初始位置是多少；
        哪些 joint 是 wheel；
        哪些 joint 是 steering；
        wheel 用 velocity actuator；
        steering 用 position actuator。