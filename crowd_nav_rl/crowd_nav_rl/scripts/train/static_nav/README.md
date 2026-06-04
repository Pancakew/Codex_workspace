# static_nav training

无障碍导航训练。

- `train_static_nav_bc.py`：用手写专家做行为克隆。
- `train_static_nav_ppo_finetune.py`：从 BC 初始化后做 PPO 微调。
- `train_static_nav.py`：从零 PPO smoke training，当前不是主线。

学习目标：让 FW-mini 根据目标相对位置输出速度/转向动作，到达随机目标。

