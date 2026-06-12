# observations

这里放“输入向量契约”，用于对齐 Isaac 训练和未来 ROS 真车 policy_node。

当前实现：

- `fwmini_obs_contract.py`
  - `main_scan44`：36 维前向主 `/scan` + 8 维 waypoint/velocity/last-command。
  - `multi_scan152`：未来多传感器输入预留切片，本轮不训练、不实现环境。

这个目录不依赖 IsaacLab，也不依赖 ROS。

