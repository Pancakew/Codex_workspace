# Scripts directory guide

这个目录只放“可直接运行的入口脚本”。环境、资产、controller 等核心逻辑不要放在这里。

## 目录结构

- `verify/`
  - 验证 FW-mini 资产、关节顺序、底盘控制链路、最小 DirectRLEnv 是否能启动。
  - 这些脚本不承担训练任务。

- `debug/static_nav/`
  - 无障碍导航环境的人工策略调试脚本。
  - 用来检查 observation、reward、done、reset 是否工作。

- `debug/obstacle_nav/`
  - 障碍物导航环境的人工策略和几何调试脚本。
  - 用来检查 obstacle clearance、collision、waypoint 专家策略、固定目标和随机目标是否合理。

- `train/static_nav/`
  - 无障碍导航训练脚本。
  - 包含从零 PPO、BC、BC 初始化 PPO 微调。

- `train/obstacle_nav/`
  - 障碍物导航训练脚本。
  - 包含 obstacle BC、waypoint-conditioned BC、BC 初始化 PPO 微调。

- `play/static_nav/`
  - 无障碍导航 checkpoint 播放和评估脚本。

- `play/obstacle_nav/`
  - 障碍物导航 checkpoint 播放和评估脚本。

## 推荐阅读顺序

1. `verify/run_verify_cmdvel.py`
   - 只验证 `v_cmd / steer_cmd -> 轮速和转向角 -> 小车运动`。

2. `verify/run_verify_env.py`
   - 验证 FW-mini DirectRLEnv 是否能 reset / step / 输出 observation。

3. `debug/static_nav/run_nav_static_vec_env.py`
   - 检查无障碍随机目标导航环境。

4. `debug/obstacle_nav/run_nav_obstacle_clearance_probe.py`
   - 不跑 Isaac 仿真，只检查障碍物几何和 clearance 计算思路。

5. `debug/obstacle_nav/run_nav_obstacle_fixed_vec_debug.py`
   - 固定目标 + 固定上绕 waypoint，适合定位单一 obstacle 任务是否可通。

6. `debug/obstacle_nav/run_nav_obstacle_random_vec_debug.py`
   - 随机目标 + waypoint 专家，接近训练数据采集场景。

7. `train/obstacle_nav/train_nav_obstacle_bc_waypoint_conditioned.py`
   - 当前 obstacle BC 主线：21 维环境 observation + 6 维 waypoint feature。

8. `train/obstacle_nav/train_nav_obstacle_ppo_finetune.py`
   - 当前 obstacle PPO 微调主线：加载 waypoint-conditioned BC checkpoint。

## cfg.py 和 env.py 的区别

以 `tasks/locomotion/nav_obstacle/` 为例：

- `fwmini_nav_obstacle_vec_env_cfg.py`
  - 只放配置，不执行仿真逻辑。
  - 定义环境参数：`num_envs`、仿真步长、episode 时长、action/observation 维度、机器人几何、目标范围、障碍物尺寸、安全阈值、reward 权重等。
  - 可以理解成“环境说明书”或“参数表”。

- `fwmini_nav_obstacle_vec_env.py`
  - 放真正的环境实现。
  - 负责创建地面、灯光、FW-mini、障碍物。
  - 负责 reset、采样目标、计算 observation、把 action 转成底层关节目标、计算 reward、判断 done。
  - 可以理解成“环境机器本体”。

简单说：`cfg.py` 说明“这个环境应该长什么样、参数是多少”；`env.py` 实现“这个环境每一步怎么运行”。

## 当前 obstacle observation / action 摘要

- 环境原始 observation 是 21 维。
- waypoint-conditioned 训练会额外拼 6 维 waypoint feature，所以网络输入是 27 维。
- action 是 2 维，当前语义是 `[forward_speed_action, steer_action]`，不是严格的 `[v, w]`。
- env 内部会把 action 缩放为 `v_cmd` 和 `steer_cmd`，再转换成 4 个轮速目标和 4 个转向角目标。
