# =============================================================================
# rsl_rl_ppo_cfg.py —— COD 站立任务的 PPO 超参数配置
#
# 【这个文件是什么】
#   IsaacLab 用 rsl_rl 库跑 PPO。这个文件定义"怎么训练"：
#     · 网络多大 (policy)
#     · PPO 超参 (algorithm)
#     · 训练规模 (runner)
#
# 【使用方式】
#   训练脚本通过 gym spec 里的 "rsl_rl_cfg_entry_point"
#   找到这里的 CodStandPPORunnerCfg，实例化后交给 OnPolicyRunner。
# =============================================================================

from isaaclab.utils import configclass

# rsl_rl 在 IsaacLab 里的封装配置类
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,     # 顶层 runner 配置
    RslRlPpoActorCriticCfg,     # actor/critic 网络结构
    RslRlPpoAlgorithmCfg,       # PPO 算法超参
)


@configclass
class CodStandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """COD 站立任务 PPO runner 配置。"""

    # ---- 训练规模与随机种子 ----
    num_steps_per_env = 24      # 每个环境每次更新采集多少步（一批 = num_envs × 该值）
    max_iterations = 1500       # 总训练迭代数（一次迭代 = 采集 + 更新）
    save_interval = 50          # 每 50 次迭代存一次 checkpoint
    experiment_name = "cod_stand"  # 实验名（日志/权重的文件夹名）
    run_name = ""               # 运行名（留空则用时间戳）
    logger = "tensorboard"      # 日志后端（tensorboard 最方便看曲线）

    # ---- 观测分组映射 ----
    #   rsl_rl 需要知道"哪个观测组喂给 policy/critic"。
    #   我们环境只有一个 obs group 叫 "policy"，所以两者都用它。
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # ---- 策略网络 (actor-critic) ----
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,             # 动作探索噪声的初始标准差（越大越爱探索）
        noise_std_type="log",           # ★ 用 log 参数化：std=exp(log_std) 恒正，避免训练中 std 变负崩溃
        actor_hidden_dims=[128, 128, 64],   # actor 网络隐藏层（3 层 MLP）
        critic_hidden_dims=[128, 128, 64],  # critic 网络隐藏层
        activation="elu",               # 激活函数（elu 是 IsaacLab 常用默认）
        actor_obs_normalization=True,   # 对 actor 的观测做归一化（训练更稳）
        critic_obs_normalization=True,  # 对 critic 的观测做归一化
    )

    # ---- PPO 算法超参 ----
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,      # 每批数据训练几遍
        num_mini_batches=4,         # 每遍切成几个 mini-batch
        learning_rate=3.0e-4,       # 学习率（1e-3→3e-4，降低以抑制训练后期发散）
        schedule="adaptive",        # 学习率调度：按 KL 自适应调整（PPO 经典做法）
        gamma=0.99,                 # 折扣因子（越接近1越重视长期回报）
        lam=0.95,                   # GAE 的 lambda（偏差-方差权衡）
        entropy_coef=0.005,         # 熵奖励系数（鼓励探索，越大越随机）
        desired_kl=0.01,            # 目标 KL（adaptive 调度用它控制步长）
        max_grad_norm=1.0,          # 梯度裁剪上限（防梯度爆炸）
        value_loss_coef=1.0,        # 价值损失权重
        use_clipped_value_loss=True,# 使用 clip 的价值损失（PPO 标配）
        clip_param=0.2,             # PPO 的 clip 范围（策略更新幅度限制）
        normalize_advantage_per_mini_batch=False,
    )
