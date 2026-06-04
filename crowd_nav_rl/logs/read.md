fwmini_static_bc：
    保存 BC policy。
    目前最重要的是 latest.pt。

fwmini_static_ppo：
    从零 PPO 的结果，效果差，暂时不作为主线。

fwmini_static_ppo_bc：
    BC 初始化 PPO 微调结果。
    play 成功率 1.0，可以作为无障碍导航策略成果。

1. logs/fwmini_static_bc/

这是 BC 训练结果。

里面：

latest.pt
bc_policy.pt

它保存的是：

无障碍随机目标导航的行为克隆策略

意义：

证明神经网络可以模仿专家策略，稳定完成无障碍目标导航。

这是你第一个真正可用的神经网络 policy。

2. logs/fwmini_static_ppo/

这是从零 PPO 的结果。

它的意义：

作为失败对照。

它说明：

直接从随机初始化 PPO 学导航，效果差，探索效率低。

所以它不是主线，不建议继续用。

3. logs/fwmini_static_ppo_bc/

这是 BC 初始化 PPO 微调结果。

意义：

在 BC 成功策略基础上，用 PPO 微调得到的无障碍导航策略。

你测试结果是：

success_rate = 1.000

所以它可以作为：

无障碍导航阶段的最终成果 checkpoint

但注意：它现在只能用于无障碍目标导航，还不能避障。障碍物环境 observation 是 21 维，而它的输入是 18 维。