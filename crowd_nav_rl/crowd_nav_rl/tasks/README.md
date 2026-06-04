# tasks

这里放 IsaacLab 任务定义。

当前重点：

- `locomotion/verify/`：最小验证环境。
- `locomotion/nav_static_vec/`：无障碍随机目标导航。
- `locomotion/nav_obstacle/`：静态障碍物导航，当前主线。
- `agents/`：旧的 rl_games 配置。

任务目录负责定义环境本身：scene、observation、reward、done、reset、action 应用。

