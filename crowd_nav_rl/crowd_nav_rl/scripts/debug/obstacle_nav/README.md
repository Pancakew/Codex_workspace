# obstacle_nav debug

障碍物导航调试脚本。

- `run_nav_obstacle_clearance_probe.py`：只检查几何和 clearance。
- `run_nav_obstacle_vec_env.py`：检查 obstacle env 的基本 observation/reward/done。
- `run_nav_obstacle_single_debug.py`：单环境固定目标固定 waypoint。
- `run_nav_obstacle_fixed_vec_debug.py`：多环境固定目标固定 waypoint。
- `run_nav_obstacle_random_vec_debug.py`：多环境随机目标 waypoint。
- `run_nav_obstacle_waypoint_env.py`：更完整的 waypoint 专家验证。
- `run_nav_obstacle_real_scanlike_debug.py`：验证 44 维 `main_scan44` scan-like observation。

训练前应该先让这些 debug 脚本证明任务是可通过的。
