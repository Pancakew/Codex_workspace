tasks/locomotion/verify/
    作用：最小验证环境。
    验证小车能不能在 DirectRLEnv 里 reset、step、接受 action、输出 observation。   

tasks/locomotion/nav_static/
    作用：单环境无障碍随机目标导航。

tasks/locomotion/nav_static_vec/
    作用：多环境并行无障碍随机目标导航。
    这是BC、PPO、PPO fine-tune 成功的主环境。
    之前已经验证过：
        num_envs = 512
        success_rate = 1.000
    说明这个环境已经可靠

tasks/locomotion/nav_obstacle/
    作用：静态障碍物导航环境。
    fwmini_nav_obstacle_vec_env_cfg.py：
        放参数：
            障碍物位置
            障碍物尺寸
            goal 范围
            robot_radius
            collision margin
            reward 参数

    fwmini_nav_obstacle_vec_env.py：
        放环境逻辑：
            reset
            action → v_cmd/steer_cmd
            v_cmd/steer_cmd → wheel/steer target
            observation
            reward
            collision
            done reason
