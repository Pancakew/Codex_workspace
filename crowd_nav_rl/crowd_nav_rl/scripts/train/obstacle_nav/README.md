# obstacle_nav training

障碍物导航训练，当前主线。

- `train_nav_obstacle_bc.py`：21 维 env observation 到 action 的 BC。
- `train_nav_obstacle_bc_waypoint_conditioned.py`：21 维 env observation + 6 维 waypoint feature 到 action 的 BC。
- `train_nav_obstacle_ppo_finetune.py`：从 waypoint-conditioned BC 初始化，用 PPO 根据 reward 微调。

当前更推荐 waypoint-conditioned 版本，因为它把“绕障碍物的阶段和临时目标”显式给到网络。

