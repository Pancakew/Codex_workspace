# logs

这里保存训练输出的 checkpoint。

重要目录：

- `fwmini_static_bc/`：无障碍 BC。
- `fwmini_static_ppo/`：无障碍从零 PPO。
- `fwmini_static_ppo_bc/`：无障碍 BC 初始化 PPO。
- `fwmini_obstacle_bc/`：障碍物 BC。
- `fwmini_obstacle_bc_wp/`：waypoint-conditioned 障碍物 BC。
- `fwmini_obstacle_ppo_bc_wp/`：waypoint-conditioned BC 初始化 PPO。
- `milestones/`：手动保存的重要阶段模型。

注意：如果修改 observation/action 语义，旧 checkpoint 可能不再兼容。

