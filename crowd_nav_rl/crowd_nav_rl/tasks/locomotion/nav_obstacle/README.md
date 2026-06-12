# nav_obstacle

这是当前最重要的环境：FW-mini 静态障碍物导航。

文件：

- `fwmini_nav_obstacle_vec_env_cfg.py`：配置表，定义 action/observation 维度、目标范围、障碍物、reward 权重、安全阈值。
- `fwmini_nav_obstacle_vec_env.py`：环境实现，定义 scene、reset、observation、action 应用、reward、done。
- `fwmini_nav_obstacle_real_scanlike_vec_env_cfg.py`：scan-like 输入对齐环境配置，observation_space 是 44。
- `fwmini_nav_obstacle_real_scanlike_vec_env.py`：复用旧环境动力学/reward/done，只把 policy observation 换成 `main_scan44`。

当前 observation 是 21 维。训练脚本可以额外拼 6 维 waypoint feature，形成 27 维网络输入。

当前 action 是 2 维 `[forward_speed_action, steer_action]`，不是严格 `[v, w]`。

`real_scanlike` 版本用于未来 ROS `/scan` 输入对齐：它不会暴露 obstacle_body_x/y/clearance，只通过 36 维 pseudo scan 表示障碍物。
