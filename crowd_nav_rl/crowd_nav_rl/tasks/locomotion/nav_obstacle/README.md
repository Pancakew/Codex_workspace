# nav_obstacle

这是当前最重要的环境：FW-mini 静态障碍物导航。

文件：

- `fwmini_nav_obstacle_vec_env_cfg.py`：配置表，定义 action/observation 维度、目标范围、障碍物、reward 权重、安全阈值。
- `fwmini_nav_obstacle_vec_env.py`：环境实现，定义 scene、reset、observation、action 应用、reward、done。

当前 observation 是 21 维。训练脚本可以额外拼 6 维 waypoint feature，形成 27 维网络输入。

当前 action 是 2 维 `[forward_speed_action, steer_action]`，不是严格 `[v, w]`。

