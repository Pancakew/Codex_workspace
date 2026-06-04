import gymnasium as gym

print("\n[System] 成功触发自定义环境注册表！正在向 Gymnasium 注入 CrowdNav...")

# 向系统正式注册你的环境图纸
# 负责将 crowd_env_cfg.py 中定义的配置打包，并注册到 Gymnasium 的环境中（例如命名为 CrowdNav-TurtleBot3-v0），使得外部训练脚本可以调用。
gym.register(
    id="CrowdNav-TurtleBot3-v0",  
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    # 告诉系统：用 NVIDIA 官方写好的 ManagerBasedRLEnv 类来生成环境。
    # 这个官方类内部已经写好了庞大且严密的 step() 和 reset() 函数。它会去自动读取 crowd_env_cfg.py，把奖励函数、观测函数“塞进”它的 step() 循环里执行。
    # 相当于在官方提供的空壳引擎中填入了你的灵魂规则。
    disable_env_checker=True,
    kwargs={ 
        # 关键字参数
        # 字符串路径格式，"模块路径:类名"
        # 配置类的路径，获取crowd_env_cfg.py里面的CrowdNavEnvCfg类
        "env_cfg_entry_point": f"{__name__}.crowd_env_cfg:CrowdNavEnvCfg", 
        # 告诉官方训练脚本去哪里找 PPO 配置文件 (已修正键名中的多余空格)
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo.yaml",

        # 这俩个字符串交给 train.py。
        # 随后，train.py 利用 Python 的反射机制和加载器，顺着这两个路径，精准地找到了配置类和 YAML 文件。
    },
)

# ==========================================
# 注册 V1 版本 (新的版)
# ==========================================
gym.register(
    id="CrowdNav-TurtleBot3-v1",  # 终端输入的新 ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={ 
        # 指向新建的 crowd_env_cfg_1.py 文件中的类
        "env_cfg_entry_point": f"{__name__}.crowd_env_cfg_1:CrowdNavEnvCfg_1", 
        
        # 算法配置可以继续复用同一个，也可以指向一个新的 yaml 文件
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_1.yaml",
    },
)


# ==========================================
# 注册 FW-mini 物理验证版本 (Sim-to-Real 底盘隔离测试)
# ==========================================
gym.register(
    id="CrowdNav-FWMini-v0",  
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={ 
        # 【修正此处】：将结尾的 CrowdEnvCfg_2 改为 CrowdNavEnvCfg_2
        "env_cfg_entry_point": f"{__name__}.crowd_env_cfg_2:CrowdNavEnvCfg_2", 
        
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_2.yaml",
    },
)