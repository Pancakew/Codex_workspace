# assets

这里放机器人资产。

当前重点是 `fw_mini/`：

- `config/fw_mini_cfg.py`：IsaacLab 加载 FW-mini USD 的配置。
- `source/`：原始 URDF/xacro/mesh。
- `usd/`：IsaacLab 实际加载的 USD 资产。

资产目录只描述机器人本体，不定义 reward、observation 或训练算法。

