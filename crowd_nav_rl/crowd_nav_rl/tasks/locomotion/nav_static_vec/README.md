# nav_static_vec

向量化无障碍随机目标导航环境。

它是 obstacle 环境的基础版本：

- 没有障碍物。
- observation 是 18 维。
- action 仍是 2 维 `[forward_speed_action, steer_action]`。
- reward 主要鼓励接近目标，惩罚距离、转向误差和动作大小。

这个目录用于先证明 FW-mini 可以学会基本目标导航。

