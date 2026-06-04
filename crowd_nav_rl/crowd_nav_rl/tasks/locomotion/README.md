# locomotion tasks

这里按导航任务类型组织环境：

- `verify/`：确认 FW-mini 能被 IsaacLab 环境管理。
- `nav_static/`：早期单环境无障碍导航。
- `nav_static_vec/`：向量化无障碍导航，训练静态导航的主线。
- `nav_obstacle/`：向量化静态障碍物导航，当前重点。
- `nav_dynamic/`、`nav_multimap/`：预留/未完成方向。
- `mdp/`：ManagerBasedRLEnv 风格的 MDP 组件预留，目前基本为空。

如果要找 reward/done/observation，请优先进入具体环境目录，例如 `nav_obstacle/`。

