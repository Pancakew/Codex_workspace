# train scripts

训练脚本目录。

- `static_nav/`：无障碍导航训练。
- `obstacle_nav/`：障碍物导航训练。

当前推荐路线：

1. 先训练/验证 static BC 或 PPO。
2. 再训练 obstacle waypoint-conditioned BC。
3. 最后用 obstacle PPO fine-tune。

