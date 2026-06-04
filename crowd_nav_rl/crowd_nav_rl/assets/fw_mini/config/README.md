# FW-mini config

这里定义 IsaacLab 如何加载 FW-mini 机器人。

主要文件：

- `fw_mini_cfg.py`

它配置：

- USD 文件路径。
- 刚体和 articulation 参数。
- wheel joint 的 velocity actuator。
- steering hinge joint 的 position actuator。

注意：这里不是 controller，也不是 RL 环境。它只告诉 IsaacLab “机器人资产长什么样、关节怎么被驱动”。

