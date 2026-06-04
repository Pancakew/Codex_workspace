# fw_mini USD

主要文件：

- `fw_mini.usd`：仿真加载的 FW-mini 主资产。
- `configuration/`：拆分出的 robot/physics/sensor/base 配置 USD。

如果机器人模型、碰撞体、关节名发生变化，需要同步检查 controller 和 env 中的 joint name 顺序。

