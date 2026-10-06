# =============================================================================
# cod_stand_env_cfg.py —— COD 平衡机器人 平地站立任务（教学用极简 Manager-based）
#
# 【这个文件在整条流程里的位置】
#   USD → 资产配置(ArticulationCfg) → 【环境配置(本文件)】 → 注册gym → PPO训练
#
# 【Manager-based 环境的 6 大"经理"】
#   ManagerBasedRLEnvCfg 里我们填这几块（都是 @configclass）：
#     scene       ① 场景：地面/机器人/灯光
#     observations② 观测：policy 能看到什么（本任务：关节角/角速度/重力投影等）
#     actions     ③ 动作：网络输出 → 关节目标位置
#     rewards     ④ 奖励：活着的奖励 + 站立惩罚等
#     terminations⑤ 终止：摔倒/超时
#     commands    ⑥ 命令：站立任务用固定零命令（可选，这里用 None 最简）
#     events      ⑦ 事件：reset 机器人、随机化
#
# 【本任务目标】让机器人原地站住、别摔倒。
#   这是入门最简单的任务，先把整条 Manager-based pipeline 跑通。
# =============================================================================

# ---- 标准库/数学 ----
import math
import torch                              # ← 自定义 reward 函数要用

# ---- IsaacLab 仿真 & 资产 ----
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

# ---- Manager 相关的配置类（每个 Manager 的术语 cfg）----
from isaaclab.managers import (
    EventTermCfg as EventTerm,           # 事件项：reset / 随机化
    ObservationGroupCfg as ObsGroup,     # 观测组：policy / critic
    ObservationTermCfg as ObsTerm,       # 观测项
    RewardTermCfg as RewTerm,            # 奖励项
    SceneEntityCfg,                      # 指定场景里的资产（如 robot 的某些关节/身体）
    TerminationTermCfg as DoneTerm,      # 终止项
)
from isaaclab.scene import InteractiveSceneCfg  # 场景基类
from isaaclab.utils import configclass          # 把普通类变成配置类（支持嵌套校验）

# ---- ISAACLAB 官方标准 mdp 函数集（通用、可复用）----
import isaaclab.envs.mdp as mdp

# ---- 我们自己的机器人资产（上一节写的）----
from agent_world.assets.cod_balance import COD_BALANCE_CFG
from isaaclab.utils.math import euler_xyz_from_quat

# =============================================================================
# ① 场景定义 (Scene)
# =============================================================================
@configclass
class CodStandSceneCfg(InteractiveSceneCfg):
    """场景：地面 + COD 机器人 + 灯光。"""

    # 地面（用一块大薄平板；本版本 GroundPlaneCfg 有材质绑定 bug，故用 CuboidCfg）
    ground = AssetBaseCfg(
        prim_path="/World/ground",                                   # USD 里的路径
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),                                # 100x100 的薄板
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # 静止不动
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),  # 顶面对齐 z=0
    )

    # COD 机器人本体
    #   {ENV_REGEX_NS} 会被 IsaacLab 自动替换成每个环境的命名空间，
    #   这样多环境并行训练时不会互相冲突。
    robot: ArticulationCfg = COD_BALANCE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # 灯光（headless 训练也要有，否则渲染报错）
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# =============================================================================
# ③ 动作定义 (Actions)
# =============================================================================
@configclass
class ActionsCfg:
    """动作：网络输出 → 关节目标位置（PD 控制）。

    IsaacLab 常用动作类型：
      mdp.JointPositionActionCfg → 输出是"关节期望角度"，驱动器 PD 去追
      mdp.JointVelocityActionCfg → 输出是"关节期望速度"（轮子常用）
    """
    # 腿关节：位置控制（网络输出每个腿关节的目标角度）
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",                      # 作用在 scene.robot 上
        joint_names=[".*_front1_joint", ".*_rear1_joint"],  # 主动腿关节（正则匹配）
        scale=0.25,                              # 动作缩放：输出乘 0.25 当作偏移量
        use_default_offset=True,                 # 在默认角基础上加偏移（更稳）
    )
    # 轮子：速度控制（网络输出每个轮子的目标速度）
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint"],
        scale=5.0,                               # 动作缩放：输出乘 5.0 (rad/s)
        use_default_offset=True,
    )


# =============================================================================
# ② 观测定义 (Observations)
# =============================================================================
@configclass
class ObservationsCfg:
    """观测：分成 policy(actor 用) 和 critic(价值网络用)。

    本任务里两者先写一样（教学最简）。后续可以给 critic 更多特权信息。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络(actor)观测。"""
        # 机身角速度（机体系）——告诉网络身体转得多快
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # ★ 机身线速度（机体系）——让网络"感知"自己实际跑多快，
        #   配合速度命令才能更好地跟踪速度（否则只能间接推断）。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=0.5)
        # 重力在机体系投影 —— 等价于"身体倾角"，最关键的姿态信息
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # ★ 速度命令：告诉网络"你被要求跑多快"（vx, vy, wz）
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 所有关节的位置（相对默认角）
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # 所有关节的速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # 上一帧动作（网络需要知道自己上一步干了啥）
        last_action = ObsTerm(func=mdp.last_action)
        # ★ 机身高度：让策略能"感知"自己离地多高，才能主动控制高度（防趴地）
        base_height = ObsTerm(func=mdp.base_pos_z)

        def __post_init__(self):
            # 观测裁剪：把各分量限制在合理范围内，避免异常值
            self.enable_corruption = False  # 先不开噪声（后续域随机化再加）
            self.concatenate_terms = True   # 把所有观测项拼成一个向量

    # 组装：policy 组。critic 组先省略（PPO 会用 policy 的）
    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# ⑥ 命令定义 (Commands) —— ★ 速度跟踪的核心
# =============================================================================
@configclass
class CommandsCfg:
    """命令：告诉机器人"应该以多快的速度前进/转向"。

    我们用 UniformVelocityCommand：每次 reset 从给定范围里**随机采样**一个
    目标速度 (vx, vy, wz) 下达给机器人，让它去跟踪。
    这样训练出的策略能跟踪**任意速度**，而不是只记住某一个 3m/s。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        # ★ 必填：每隔多久(秒)重新随机采样一次速度命令
        resampling_time_range=(10.0, 10.0),
        # 采样范围：x 方向 0.0~3.0 m/s（只让它往前跑），y/z 先固定 0
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.0),   # ← 前进速度范围 0~3 m/s（想固定 3 就写 (3.0,3.0)）
            lin_vel_y=(0.0, 0.0),   # 侧向速度：先不给（=0）
            ang_vel_z=(0.0, 0.0),   # 转向角速度：先不给（=0）
        ),
        # 让少量环境(10%)保持静止，训练鲁棒性（可选）
        rel_standing_envs=0.1,
        # 用 heading 命令（这里先关，直接用角速度）
        heading_command=False,
    )


# =============================================================================
# ④ 奖励定义 (Rewards)
# =============================================================================
def body_roll_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    
    gravity_b = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.square(gravity_b[:, 1])  
 
def body_pitch_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

    gravity_b = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.square(gravity_b[:, 0])   

def body_piitch_l2_reward(env,target_pitch=0.0,std=0.05,asset_cfg=SceneEntityCfg("robot")):
    
    gravity_b = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.exp(-torch.square(gravity_b[:, 0] - target_pitch) / (std ** 2))
    
def yaw_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """惩罚偏航角速度 wz（转圈/绕圆）。

    用角速度而非绝对角度：角速度连续、平滑、无 ±pi 跳变，安全且能直接禁止"转圈"。
    """
    return torch.square(env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2])

def base_vy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """惩罚机身侧向线速度 vy（走直线就不该有侧向速度）。

    root_lin_vel_b[:, 1] 是机体系左右方向线速度。直行时应≈0，
    绕圆/侧滑时会变大，惩罚它能逼机器人走直线。
    """
    return torch.square(env.scene[asset_cfg.name].data.root_lin_vel_b[:, 1])

def base_height_exp(env, target_height=0.25, std=0.05,
                    asset_cfg=SceneEntityCfg("robot")):
    """高度指数奖励：越接近 target_height 越接近 1（配正权重）。"""
    h = env.scene[asset_cfg.name].data.root_pos_w[:, 2]  # 机身世界 z (N,)
    # ---- [DEBUG] 打印机身高度：每 N 次调用打印一次（临时调试用，定位高度问题后删）----
    if not hasattr(env, "_dbg_height_cnt"):
        env._dbg_height_cnt = 0
    env._dbg_height_cnt += 1
    if env._dbg_height_cnt % 200 == 0:
        print(f"[DEBUG-HEIGHT] mean={h.mean().item():.4f}  min={h.min().item():.4f}  "
              f"max={h.max().item():.4f}  target={target_height}")
    return torch.exp(-torch.square(h - target_height) / (std ** 2))

@configclass
class RewardsCfg:
    """奖励：鼓励"站着且直立"，惩罚"抖动/乱动/趴地"。"""

    # 活着就给奖励（每步 +1），鼓励坚持不摔
    alive = RewTerm(func=mdp.is_alive, weight=5.0)

    # ★★ 速度跟踪奖励（本任务核心）：
    #   track_lin_vel_xy_exp 用指数核 exp(-误差/std²) 衡量 x/y 线速度跟踪好坏。
    #   误差越小 → 奖励越接近 1；误差越大 → 接近 0。
    #   std 越小，对速度误差越"苛刻"。
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,                       # 权重给大，这是主要学习目标
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )

    # ★ 机身高度奖励：让机身尽量贴近目标高度 0.25m。
    #   base_height_l2 返回 (当前高度 - 目标高度)^2，用负权重惩罚偏离。
    #   这一项是修复"趴地上躺平"的关键：躺平高度≈0.05m，偏差大 → 惩罚大，
    #   迫使机器人主动把身体"撑起来"。
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,                
        params={"target_height": 0.28},
   )
    height_track = RewTerm(
        func=base_height_exp,
        weight=10.0,                             # ★ 正权重
        params={"target_height": 0.28, "std": 0.05},
    )

    body_roll = RewTerm(func=body_roll_l2, weight=-8.0)
    body_pitch = RewTerm(func=body_pitch_l2, weight=-3.0)
    body_pitch_re=RewTerm(func= body_piitch_l2_reward,weight=3.0,params={"target_pitch":0.0,"std": 0.15},)
    # ★ 转向角速度跟踪（命令 wz=0）→ 鼓励"不转"，配合下面 yaw_rate 一起压制转圈
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.15)},
    )
    # ★ 惩罚偏航角速度 wz（治"原地转圈s/绕圆"）
    yaw_rate = RewTerm(func=yaw_rate_l2, weight=-0.6)
    # ★ 惩罚侧向线速度 vy（治"歪着走/绕圆"，逼它走直线）
   # body_vy = RewTerm(func=base_vy_l2, weight=-2.0)

    # 惩罚机身竖直速度（减少上下颠簸）
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)

    # 惩罚动作变化率（让动作平滑、不抖）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    # 惩罚关节速度（减少乱摆）
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)

    # 惩罚关节力矩（省电、保护电机）
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)


# =============================================================================
# ⑤ 终止定义 (Terminations)
# =============================================================================
@configclass
class TerminationsCfg:
    """终止：什么时候算"这一局结束"并 reset。"""

    # 超时（达到 episode 最大步数，正常结束）
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 机身过低（快摔了/趴地了）→ 判定失败
    #   低于 base_link 高度 < 0.15m 就终止。
    #   提到 0.15（原来 0.12）→ 让"趴地躺平"立刻结束，绝不允许靠躺平混奖励。
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.17},
    )

    # 机身倾斜过大（翻了）→ 终止
    #   projected_gravity 的 z 分量 > -0.5（即倾斜超过约 60°）
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )


# =============================================================================
# ⑦ 事件定义 (Events)
# =============================================================================
@configclass
class EventCfg:
    """事件：reset 时做什么、训练中做哪些随机化。"""

    # reset 时把机器人放回初始姿态（位置+关节角）
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)},
            "velocity_range": {},  # 初速度保持 0
        },
    )

    # reset 时给关节一点随机初始角（鲁棒性）
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),  # 在默认角基础上 ±0.1 rad
            "velocity_range": (0.0, 0.0),
        },
    )


# =============================================================================
# 根配置：把所有 Manager 组装成一个 EnvCfg
# =============================================================================
@configclass
class CodStandEnvCfg(ManagerBasedRLEnvCfg):
    """COD 平地站立环境总配置。"""

    # 各 Manager（按需提供，没提供的用 IsaacLab 默认）
    scene: CodStandSceneCfg = CodStandSceneCfg(num_envs=4096, env_spacing=2.5)
    # 说明：num_envs 默认给 4096（正式训练用）；本地 8GB 显存测试时
    #      用 parse_env_cfg(..., num_envs=64) 覆盖成小值。
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()   # ★★★★★ 这一行是必须的！挂上速度命令

    def __post_init__(self):
        """初始化后处理：设置仿真步长、episode 长度、decimation 等。"""
        # ---- 仿真基础参数 ----
        self.decimation = 4          # 每 4 个物理步 → 1 个策略步（控制频率降 4 倍）
        self.episode_length_s = 10.0 # 每个 episode 10 秒

        # ---- 仿真器设置 ----
        self.sim.dt = 1.0 / 200.0    # 物理步长 = 5ms（200Hz）
        self.sim.render_interval = self.decimation
        self.sim.device = "cuda:0"
        self.sim.use_fabric = False  # 走 USD 读写，避免本版本 fabric 下 DOF 读取失败

        # ---- 视图相机（play 可视化用）----
        self.viewer.eye = (3.0, 0.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)
