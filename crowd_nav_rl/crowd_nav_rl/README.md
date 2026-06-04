# Python 包目录

这里是 `crowd_nav_rl` 的 Python 包源码。

主要子目录：

- `assets/`：FW-mini 机器人资产配置和 USD/URDF 源文件。
- `controllers/`：独立底盘控制器。
- `tasks/`：IsaacLab 任务/环境。
- `scripts/`：命令行入口脚本。
- `utils/`：预留工具函数，目前大多为空。

一般读代码时，先看 `tasks/locomotion/nav_obstacle/`，再看 `scripts/train/obstacle_nav/`。

