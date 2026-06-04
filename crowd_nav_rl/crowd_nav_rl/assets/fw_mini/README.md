# FW-mini 资产

这个目录保存 FW-mini 小车模型。

你需要区分：

- `source/`：机器人原始描述文件，便于追溯模型来源。
- `usd/`：IsaacLab 仿真运行时加载的文件。
- `config/`：把 USD 包装成 IsaacLab `ArticulationCfg`。

导航环境通过 `config/fw_mini_cfg.py` 导入 `FW_MINI_CFG`。

