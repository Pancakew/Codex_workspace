# crowd_nav_rl 项目说明

这个项目的目标是在 IsaacLab 中训练 FW-mini 小车完成导航任务：

1. 先验证 FW-mini 资产和底盘控制是否能正常运动。
2. 再做无障碍目标导航。
3. 最后加入静态障碍物，用 waypoint 专家生成示范数据，训练策略避障导航。

当前主要训练路线是：

`verify -> debug/static_nav -> train/static_nav -> debug/obstacle_nav -> train/obstacle_nav -> play`

核心概念：

- `assets/`：机器人模型和 IsaacLab asset 配置。
- `controllers/`：底盘命令到轮速/转向角的转换。
- `tasks/`：IsaacLab 环境定义，包含 observation、reward、done、reset。
- `scripts/`：可以直接运行的验证、调试、训练、播放入口。
- `logs/`：训练出的 checkpoint。

