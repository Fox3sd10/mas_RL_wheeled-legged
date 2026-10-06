# =============================================================================
# cod_stand：COD 平衡机器人 —— 平地站立任务（教学用极简 Manager-based）
# =============================================================================
# 这个 __init__.py 负责用 gym.register 把任务注册进 gym，
# 这样训练脚本就能用 gym.make("Cod-Stand-v0") 找到它。

import gymnasium as gym

from . import agents  # noqa: F401  导入 agents 包，供下面用 agents.__name__ 拼路径

##
# 注册 gym 环境
##
gym.register(
    id="Cod-Stand-v0",  # 任务唯一 ID（训练/play 时用这个字符串）
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 使用 IsaacLab 标准 Manager 环境
    disable_env_checker=True,
    kwargs={
        # ① 环境配置类入口："包路径.模块名:类名"
        "env_cfg_entry_point": f"{__name__}.cod_stand_env_cfg:CodStandEnvCfg",
        # ② PPO 训练配置入口（训练脚本用 "--task" 时读这个 key）
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CodStandPPORunnerCfg",
    },
)
