# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

# 导入 OrderedDict：有序字典，保证奖励项等键值对顺序稳定
from collections import OrderedDict
# 导入 copy：深拷贝用，避免多个 cfg 共享同一可变对象被互相串改
import copy
# 导入 torch：张量库，这里主要用来把角度（度）转成弧度（torch.pi）
import torch

# 本仓库自定义的 mdp 集合（封装 IsaacLab manager 的常用 mdp 函数、地形生成器等）
import agent_tasks.manager.mdp.isaaclab as mdp
# IsaacLab 仿真工具（这里用到 RigidBodyMaterialCfg 物理材质）
import isaaclab.sim as sim_utils
# 光线投射传感器配置及其光线分布图案（GridPatternCfg 网格图案）
from isaaclab.sensors import RayCasterCfg, patterns
# 地形导入器配置（plane / generator 两种地形类型）
from isaaclab.terrains import TerrainImporterCfg
# 地形级命令覆盖配置：针对指定地形覆盖速度/角速度命令范围与 reset 行为
from agent_tasks.manager.mdp.terrain import TerrainCommandOverrideCfg


# 任务标志（task flag）观测维度：预留 1 维占位，紧邻 command 观测
V14_TASK_FLAG_DIM = 1
# 基础 policy（actor）观测维度 = 28
V14_BASE_POLICY_OBS_DIM = 28
# 基础 privileged（critic）观测维度 = 32
V14_BASE_PRIVILEGED_OBS_DIM = 32
# 在基础观测之外额外拼上的观测维度（85 本体感觉 + 44 特权信息）
V14_PRIV_OBS_PLUS_EXTRA_DIM = 85 + 44
# 带额外信息的 privileged 观测总维度 = 基础特权维 + 额外维
V14_PRIV_OBS_PLUS_PRIVILEGED_DIM = V14_BASE_PRIVILEGED_OBS_DIM + V14_PRIV_OBS_PLUS_EXTRA_DIM
# 关节 sin/cos 编码额外增加 6 维
V14_JOINT_SINCOS_EXTRA_DIM = 6
# 关节 sin/cos 编码后的 policy 观测维度 = 基础 policy 维 + 6
V14_JOINT_SINCOS_POLICY_OBS_DIM = V14_BASE_POLICY_OBS_DIM + V14_JOINT_SINCOS_EXTRA_DIM
# 关节 sin/cos 编码后的 privileged 观测维度 = 基础特权维 + 6
V14_JOINT_SINCOS_PRIVILEGED_OBS_DIM = V14_BASE_PRIVILEGED_OBS_DIM + V14_JOINT_SINCOS_EXTRA_DIM
# 关节 sin/cos 动作编码：动作维度 = 10
V14_JOINT_SINCOS_ACT_SINCOS_ACTION_DIM = 10
# 关节 sin/cos 动作编码额外增加 4 维
V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM = 4
# 关节 sin/cos（观测 + 动作）后的 policy 观测维度 = sin/cos 观测维 + 4
V14_JOINT_SINCOS_ACT_SINCOS_POLICY_OBS_DIM = (
    V14_JOINT_SINCOS_POLICY_OBS_DIM + V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM
)
# 关节 sin/cos（观测 + 动作）后的 privileged 观测维度 = sin/cos 特权维 + 4
V14_JOINT_SINCOS_ACT_SINCOS_PRIVILEGED_OBS_DIM = (
    V14_JOINT_SINCOS_PRIVILEGED_OBS_DIM + V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM
)
# 崎岖地形"高度偏移课程学习"的默认配置（dict，后续会被用户 cfg 覆盖更新）
V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG = {
    "enabled": False,  # 是否启用高度偏移课程学习
    "interval": 500,  # 每多少次迭代推进一级难度
    "max_iteration": 5000,  # 课程最大迭代数
    "num_levels": 11,  # 难度等级（地形行数）数量
    "steps_per_iteration": 24,  # 每个迭代对应的 env step 数
    "random_reset_up_to_current_level": False,  # 是否只在当前等级以内随机重置
    "random_reset_after_max": True,  # 达到最大迭代后是否随机重置
    "randomize_type_on_random_reset": True,  # 随机重置时是否随机化地形类型
}
# DreamWaQ 算法：估计状态维度 = 4
V14_DREAMWAQ_ESTIMATED_STATE_DIM = 4
# DreamWaQ 算法：policy 历史帧数 = 5
V14_DREAMWAQ_POLICY_HIST = 5
# SEMA 算法：估计状态维度 = 6
V14_SEMA_ESTIMATED_STATE_DIM = 6
# SEMA 算法：policy 历史帧数 = 10
V14_SEMA_POLICY_HIST = 10
# Residual 残差算法：估计状态维度 = 4
V14_RESIDUAL_ESTIMATED_STATE_DIM = 4
# Residual 残差算法：policy 历史帧数 = 5
V14_RESIDUAL_POLICY_HIST = 5
# HIM 算法：估计状态维度 = 4
V14_HIM_ESTIMATED_STATE_DIM = 4
# HIM 算法：policy 历史帧数 = 5
V14_HIM_POLICY_HIST = 5
# NP3O 算法：policy 历史帧数 = 10
V14_NP3O_POLICY_HIST = 10
# NP3O 算法：估计状态维度 = 4
V14_NP3O_EST_DIM = 4
# NP3O 算法：cost（约束通道）数 = 5
V14_NP3O_COST_DIM = 5
# NP3O 的 on-constraint 观测维度 = 基础特权维 + 基础 policy 维 × policy 历史帧数
V14_NP3O_ON_CONSTRAINT_DIM = (
    V14_BASE_PRIVILEGED_OBS_DIM
    + V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST
)
# NP3O 的 on-constraint 额外特权观测维度 = 基础特权维 + 额外维 + 基础 policy 维 × policy 历史帧数
V14_NP3O_ON_CONSTRAINT_EXTRA_PRIV_DIM = (
    V14_BASE_PRIVILEGED_OBS_DIM
    + V14_PRIV_OBS_PLUS_EXTRA_DIM
    + V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST
)
# 机身高度扫描器：网格尺寸（x=y=0.02m）
V14_BODY_HEIGHT_SCANNER_GRID_SIZE = (0.02, 0.02)
# 机身高度扫描器：光线分辨率 0.01m
V14_BODY_HEIGHT_SCANNER_RESOLUTION = 0.01
# 轮子高度扫描器：网格尺寸（x=y=0.015m）
V14_WHEEL_HEIGHT_SCANNER_GRID_SIZE = (0.015, 0.015)
# 轮子高度扫描器：光线分辨率 0.01m
V14_WHEEL_HEIGHT_SCANNER_RESOLUTION = 0.01

# 基础观测的裁剪（clip）上下限：键=观测项名，值=标量(±限制) 或 [lower, upper] 区间
V14_BASIC_OBS_CLIP: dict = {
    "command": 100.0,  # 命令（速度/角速度）裁剪 ±100
    "height_cmd": [0.0, 1.0],  # 高度命令裁剪到 [0, 1]
    "root_ang_vel_b": 100.0,  # 机体系角速度裁剪 ±100
    "projected_gravity_b": 100.0,  # 机体系投影重力裁剪 ±100
    "joint_pos": 100.0,  # 关节位置裁剪 ±100
    "joint_vel_leg": 200.0,  # 腿关节速度裁剪 ±200
    "joint_vel_wheel": 200.0,  # 轮关节速度裁剪 ±200
    "actions": 100.0,  # 动作裁剪 ±100
    "root_lin_vel_b": 100.0,  # 机体系线速度裁剪 ±100
    "obs_height": [-10.0, 10.0],  # 观测高度裁剪 [-10, 10]
}

# 基础观测的缩放（scale）系数：让不同量纲的观测大致归一化
V14_BASIC_OBS_SCALE: dict = {
    "command": {  # 命令各分量缩放
        "lin_vel_x": 1.0,  # x 线速度命令缩放 1.0
        "lin_vel_y": 1.0,  # y 线速度命令缩放 1.0
        "ang_vel_z": 1.0,  # z 角速度命令缩放 1.0
    },
    "height_cmd": 5.0,  # 高度命令缩放 5.0
    "root_ang_vel_b": 0.5,  # 机体系角速度缩放 0.5
    "joint_vel_leg": 0.1,  # 腿关节速度缩放 0.1
    "joint_vel_wheel": 0.1,  # 轮关节速度缩放 0.1
    "obs_height": 5.0,  # 观测高度缩放 5.0
}

# 额外观测的裁剪配置：先深拷贝基础裁剪，再补充额外项
V14_EXTRA_OBS_CLIP = copy.deepcopy(V14_BASIC_OBS_CLIP)
V14_EXTRA_OBS_CLIP.update(
    {
        "spring_force": 1000.,  # 弹簧力裁剪 ±1000
        # 以下数项基础版里没有，如需要可打开（世界系位置/速度等）：
        # "root_pos_w": 100.0,
        # "root_lin_vel_w": 100.0,
        # "root_ang_vel_w": 200.0,
        "joint_torque": 100.0,  # 关节力矩裁剪 ±100
        "obs_delay_steps": 100.0,  # 观测延迟步数裁剪 ±100
        "act_delay_steps": 100.0,  # 动作延迟步数裁剪 ±100
        "joint_acc": 1000.0,  # 关节加速度裁剪 ±1000
        "wheel_body_lin_vel": 100.0,  # 轮体线速度裁剪 ±100
        "wheel_contact_force": 5000.0,  # 轮接触力裁剪 ±5000
        "wheel_contact_state": 1.0,  # 轮接触状态裁剪 ±1（0/1）
        "joint_stiffness": 100.0,  # 关节刚度裁剪 ±100
        "joint_damping": 100.0,  # 关节阻尼裁剪 ±100
        "joint_friction": 100.0,  # 关节摩擦裁剪 ±100
        "body_mass": 100.0,  # 机身质量裁剪 ±100
        "body_mass_scale": 10.0,  # 机身质量缩放因子裁剪 ±10
        "body_inertia_diag": 100.0,  # 机身惯量对角元裁剪 ±100
        "body_material": 100.0,  # 机身材质参数裁剪 ±100
        "body_com": 100.0,  # 机体质心裁剪 ±100
    }
)

# 额外观测的缩放配置：先深拷贝基础缩放，再补充额外项
V14_EXTRA_OBS_SCALE = copy.deepcopy(V14_BASIC_OBS_SCALE)
V14_EXTRA_OBS_SCALE.update(
    {
        "spring_force": 0.01,  # 弹簧力缩放 0.01
        # 以下数项如需要可打开：
        # "root_pos_w": 0.1,
        # "root_lin_vel_w": 0.25,
        # "root_ang_vel_w": 0.25,
        "joint_torque": 0.05,  # 关节力矩缩放 0.05
        "obs_delay_steps": 1.0,  # 观测延迟步数缩放 1.0
        "act_delay_steps": 1.0,  # 动作延迟步数缩放 1.0
        "joint_acc": 0.01,  # 关节加速度缩放 0.01
        "wheel_body_lin_vel": 1.0,  # 轮体线速度缩放 1.0
        "wheel_contact_force": 0.01,  # 轮接触力缩放 0.01
        "wheel_contact_state": 1.0,  # 轮接触状态缩放 1.0
        "joint_stiffness": 1.0,  # 关节刚度缩放 1.0
        "joint_damping": 1.0,  # 关节阻尼缩放 1.0
        "joint_friction": 1.0,  # 关节摩擦缩放 1.0
        "body_mass": 0.1,  # 机身质量缩放 0.1
        "body_mass_scale": 1.0,  # 机身质量缩放因子缩放 1.0
        "body_inertia_diag": 1.0,  # 机身惯量对角元缩放 1.0
        "body_material": 1.0,  # 机身材质缩放 1.0
        "body_com": 1.0,  # 机体质心缩放 1.0
    }
)

# 腿关节按固定顺序排列的关节名（左右、前后交错），保证索引与 IK 顺序一致
V14_ORDERED_LEG_JOINT_NAMES: tuple[str, ...] = (
    "left_front1_joint",  # 左前 1
    "right_front1_joint",  # 右前 1
    "left_rear1_joint",  # 左后 1
    "right_rear1_joint",  # 右后 1
    "left_front2_joint",  # 左前 2
    "right_front2_joint",  # 右前 2
    "left_front3_joint",  # 左前 3
    "right_front3_joint",  # 右前 3
    "left_front4_joint",  # 左前 4
    "right_front4_joint",  # 右前 4
    "left_rear2_joint",  # 左后 2
    "right_rear2_joint",  # 右后 2
)

# 腿连杆按固定顺序排列的刚体名（与关节名一一对应）
V14_ORDERED_LEG_BODY_NAMES: tuple[str, ...] = (
    "left_front1_link",  # 左前 1
    "right_front1_link",  # 右前 1
    "left_rear1_link",  # 左后 1
    "right_rear1_link",  # 右后 1
    "left_front2_link",  # 左前 2
    "right_front2_link",  # 右前 2
    "left_front3_link",  # 左前 3
    "right_front3_link",  # 右前 3
    "left_front4_link",  # 左前 4
    "right_front4_link",  # 右前 4
    "left_rear2_link",  # 左后 2
    "right_rear2_link",  # 右后 2
)

# 腿部连杆长度（单位 m），用于运动学计算（IK / 腿长估计）
V14_LINKS_LENGTH = [
    0.11340111711971801,  # 第一节连杆长度
    0.13499721265641007,  # 第二节连杆长度
    0.21,  # 第三节（轮轴）连杆长度
]

# 关节角度零位偏置 alpha0（单位：度）
_V14_ALPHA0_DEG = 9.430885159953315
# 关节角度零位偏置 alpha2（单位：度）
_V14_ALPHA2_DEG = 37.874675469056214
# 各关节的角度偏置（弧度），把电机读数映射到几何角
V14_ALPHA_OFFSET = [
    _V14_ALPHA0_DEG / 180.0 * torch.pi,  # 关节 1 偏置（alpha0 → 弧度）
    torch.pi,  # 关节 2 偏置 = π
    _V14_ALPHA2_DEG / 180.0 * torch.pi,  # 关节 3 偏置（alpha2 → 弧度）
    (180.0 + _V14_ALPHA0_DEG - 2.0 * _V14_ALPHA2_DEG) / 180.0 * torch.pi,  # 关节 4 偏置（组合式）
    (20.0 + _V14_ALPHA2_DEG) / 180.0 * torch.pi,  # 关节 5 偏置（20° + alpha2）
    _V14_ALPHA2_DEG / 180.0 * torch.pi,  # 关节 6 偏置（alpha2 → 弧度）
]

# 地面 reset（预定义落地姿态）配置：让机器人以某种腿姿落地后自行起身
V14_PREDEFINED_RESET_GROUND = dict(
    modes={  # 多种"落地模式"，各带概率
        "positive": dict(  # 正向模式
            prob=0.3,  # 被选中概率
            sign=1.0,  # 方向符号 +1
            leg_height=[-0.06, 0.12],  # 腿高度采样范围
            leg_length=[0.14, 0.36],  # 腿长采样范围
        ),
        "negative": dict(  # 反向模式
            prob=0.2,  # 被选中概率
            sign=-1.0,  # 方向符号 -1
            leg_height=[-0.06, 0.],  # 腿高度采样范围
            leg_length=[0.14, 0.36],  # 腿长采样范围
        ),
    },
    # 向后兼容字段：仅当 modes 缺失时使用
    prob=0.2,  # 兼容用概率
    leg_height=[-0.06, 0.12],  # 兼容用腿高度范围
    leg_length=[0.14, 0.36],  # 兼容用腿长范围
    # 所有地面落地模式共享的 env 级 reset 设置
    start_reset_time=1.5,  # 开始 reset 的时间（s）
    start_root_height=0.25,  # 初始机身高度（m）
    zero_torque_time_s=0.2,  # 力矩置零的持续时间（s）
    command_ranges={  # reset 期间使用的命令范围
        "lin_vel_x": (-1.0, 1.0),  # x 线速度范围
        "lin_vel_y": (0.0, 0.0),  # y 线速度范围（固定 0）
        "ang_vel_z": (-1.0, 1.0),  # z 角速度范围
    },
)

# 旧版（legacy）观测输入缩放配置：保留以兼容老策略/老 ckpt
V14_LEGACY_OBS_INPUT_SCALE_CFG = {
    "command": 1.0,  # 命令缩放
    "height_cmd": 5.0,  # 高度命令缩放
    "root_ang_vel_b": 1.0,  # 机体系角速度缩放
    "joint_pos": 1.0,  # 关节位置缩放
    "joint_vel_leg": 0.1,  # 腿关节速度缩放
    "joint_vel_wheel": 0.1,  # 轮关节速度缩放
    "root_lin_vel_b": 1.0,  # 机体系线速度缩放
    "obs_height": 5.0,  # 观测高度缩放
    "root_lin_vel_w": 1.0,  # 世界系线速度缩放
    "root_ang_vel_w": 1.0,  # 世界系角速度缩放
    "wheel_body_lin_vel": 1.0,  # 轮体线速度缩放
    "joint_torque": 0.1,  # 关节力矩缩放
}

# 崎岖地形下，按"地形名"覆盖速度/角速度命令范围与 reset 行为的配置表
# 键=地形子名（需存在于 RM_ROUGH_TERRAINS_CFG 中），值=TerrainCommandOverrideCfg
V14_ROUGH_TERRAIN_COMMAND_OVERRIDES: dict[str, TerrainCommandOverrideCfg] = {
    # ---- 以下为历史/备选地形覆盖项，均已注释（保留供参考）----
    # "cliff_inv_stair_slope_flat_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.2, 0.4)],
    #     lin_vel_x=[(-2.5, 2.5)],
    #     lin_vel_y=(0.0, 0.0),
    # ),
    # "cliff_inv_stair_slope_for_rm1": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(1.0, 2.5), (-2.5, -1.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "cliff_inv_stair_slope_for_rm2": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(1.0, 2.5), (-2.5, -1.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "cliff_inv_stair_slope_for_rm3": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "cliff_inv_stair_slope_for_rm4": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.01, 0.01),
    #     reset_heading_axis_aligned_only=True,
    # ),
    # 高台阶地形：高度 0.22~0.34，高速前后各一段，含航向角速度命令
    "high_stair_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.34)],  # 机身高度范围
        lin_vel_x=[(1.5, 2.7), (-2.7, -1.5)],  # x 速度：前后各一段
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        ang_vel_z_heading=(-1, 1),  # 有航向时的角速度范围
        ang_vel_z_non_heading=(-0.1, 0.1),  # 非航向时的小角速度
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
    # 高速台阶地形：与高台阶类似，但注释掉航向角速度（只用非航向）
    "high_speed_stair_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.34)],  # 机身高度范围
        lin_vel_x=[(1.5, 2.7), (-2.7, -1.5)],  # x 速度：前后各一段
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # ang_vel_z_heading=(-1, 1),  # 注释掉航向角速度
        ang_vel_z_non_heading=(-0.1, 0.1),  # 非航向时的小角速度
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
    # 低速台阶地形：速度范围较小（0.5~1.5）
    "low_speed_stair_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.34)],  # 机身高度范围
        # lin_vel_x=[(2.0, 3.0), (-3.0, -2.0)],  # 原高速版（注释）
        lin_vel_x=[(0.5, 1.5), (-1.5, -0.5)],  # x 速度：低速前后各一段
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # ang_vel_z_heading=(-1, 1),  # 注释掉航向角速度
        ang_vel_z_non_heading=(-0.1, 0.1),  # 非航向时的小角速度
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
    # "inv_pyramid_stair_slope_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.40)],
    #     lin_vel_x=(-2.5, 2.5),
    #     lin_vel_y=(0.0, 0.0),
    # ),
    # "plane_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.20, 0.40)],
    #     lin_vel_x=(-0.0, 0.0),
    #     lin_vel_y=(0.0, 0.0),
    #     ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=[(-12, -3), (3, 12)],
    #     reset_heading_axis_aligned_only=True,
    # ),
    # "stair_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.37, 0.40)],
    #     lin_vel_x=[(1.0, 2.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.5, 0.5),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_special_mode=True,
    # ),
    # "inv_stair_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.37, 0.40)],
    #     lin_vel_x=[(1.0, 2.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.1, 0.1),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_special_mode=True,
    # ),
    # 'tiny_step': TerrainCommandOverrideCfg(
    #     height_range=[(0.20, 0.30)],
    #     lin_vel_x=[(1.5, 2.5), (-2.5, -1.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-1.0, 1.0),
    # ),
    # "cliff_inv_stair_slope_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.1, 0.1),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    # ),
    # "cliff_inv_stair_slope_tall_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.28, 0.35)],
    #     lin_vel_x=[(2.0, 2.5), (-2.5, -2.0)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0.1, 0.1),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_special_mode=True,
    # ),
    # 演示（play）专用：固定高度 0.3、固定前向速度 2.5、无旋转
    "cliff_inv_stair_slope_short_for_rm_play": TerrainCommandOverrideCfg(
        height_range=[(0.3, 0.3)],  # 高度固定 0.3
        lin_vel_x=[(2.5, 2.5)],  # 前向速度固定 2.5
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # ang_vel_z_heading=(-0.1, 0.1),  # 注释掉航向角速度
        ang_vel_z_non_heading=(-0., 0.),  # 非航向角速度固定 0
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_predefined_reset_ground=True,  # 禁用预定义地面 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
    # 训练用：中高速台阶（2.0~2.7）
    "cliff_inv_stair_slope_short_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],  # 机身高度范围
        lin_vel_x=[(2.0, 2.7), (-2.7, -2.0)],  # x 速度：高速前后各一段
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # ang_vel_z_heading=(-1, 1),  # 注释掉航向角速度
        ang_vel_z_non_heading=(-0.1, 0.1),  # 非航向时的小角速度
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        # disable_predefined_reset_ground=True,  # 注释掉（不禁用地面 reset）
        disable_special_mode=True,  # 禁用特殊模式
    ),
    # "cliff_inv_stair_slope_long_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.25, 0.34)],
    #     lin_vel_x=[(1.5, 2.6), (-2.6, -1.5)],
    #     lin_vel_y=(0.0, 0.0),
    #     # ang_vel_z_heading=(-1, 1),
    #     ang_vel_z_non_heading=(-0., 0.),
    #     reset_heading_axis_aligned_only=True,
    #     disable_predefined_reset_air = True,
    #     disable_predefined_reset_ground=True,
    #     disable_special_mode=True,
    # ),
    # 上坡地形（低难度）：只覆盖机身高度范围
    'slope_for_rm_low': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 上坡地形（高难度）：高度更紧，前后速度、禁用空中 reset
    'slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],  # 机身高度范围
        lin_vel_x=[(-2.5, 2.5)],  # x 速度范围（双向连续）
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # reset_heading_axis_aligned_only=True,  # 注释掉
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        # disable_special_mode=True,  # 注释掉（保留特殊模式）
    ),
    # 下坡地形（低难度）：只覆盖机身高度范围
    'inv_slope_for_rm_low': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 下坡地形（高难度）
    'inv_slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.25, 0.34)],  # 机身高度范围
        lin_vel_x=[(-2.5, 2.5)],  # x 速度范围（双向连续）
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
    # 'stair_slope_for_rm_low': TerrainCommandOverrideCfg(  # 注释掉
    #     height_range=[(0.22, 0.42)],
    # ),
    # 台阶-坡道地形（高难度）：角速度允许全向 [-π, π]
    'stair_slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],  # 机身高度范围
        lin_vel_x=[(-2.5, 2.5)],  # x 速度范围
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # ang_vel_z_heading=(-1, 1),  # 注释掉
        ang_vel_z_non_heading=(-torch.pi, torch.pi),  # 非航向角速度：全向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        # disable_special_mode=True,  # 注释掉（保留特殊模式）
    ),
    # 反台阶-坡道（低难度）
    'inv_stair_slope_for_rm_low': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
        # height_range=[(0.24, 0.38)],  # 备选范围（注释）
    ),
    # 反台阶-坡道（高难度）：高速前后各一段
    'inv_stair_slope_for_rm_high': TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.32)],  # 机身高度范围
        lin_vel_x=[(1.5, 2.7), (-2.7, -1.5)],  # x 速度：高速前后各一段
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        ang_vel_z_non_heading=(-0.1, 0.1),  # 非航向时的小角速度
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
    # 随机均匀地形：只覆盖机身高度范围
    'random_uniform_for_rm': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 小台阶地形：只禁用预定义空中 reset
    'tiny_step': TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.32)],  # 机身高度范围
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        # disable_predefined_reset_ground=True,  # 注释掉（不禁用地面 reset）
    ),
    # "fort_for_rm": TerrainCommandOverrideCfg(
    #     height_range=[(0.24, 0.4)],
    # ),
}

# 旋转类任务（小陀螺/转向）地形命令覆盖表 —— 版本 1
V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1: dict[str, TerrainCommandOverrideCfg] = {
    # 小台阶（旋转版）：只覆盖机身高度范围
    "tiny_step_rot": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 上坡地形（低难度）
    "slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 下坡地形（低难度）
    "inv_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 台阶-坡道（低难度）
    "stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 台阶-坡道（中难度）
    "stair_slope_for_rm_mid": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.38)],  # 机身高度范围
    ),
    # 反台阶-坡道（低难度）
    "inv_stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
        # height_range=[(0.24, 0.38)],  # 备选范围（注释）
    ),
    # 平面（旋转版）
    "plane_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.20, 0.42)],  # 机身高度范围
    ),
    # 随机均匀地形
    "random_uniform_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 台阶（旋转版）：低速前后各一段，非航向角速度 ±1
    "stair_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.2, 0.42)],  # 机身高度范围
        lin_vel_x=[(0.5, 2.0), (-2.0, -0.5)],  # x 速度：低速前后各一段
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # ang_vel_z_heading=(-1, 1),  # 注释掉航向角速度
        ang_vel_z_non_heading=(-1., 1.),  # 非航向角速度 ±1
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
}

# 旋转类任务（小陀螺/转向）地形命令覆盖表 —— 版本 2（比版本 1 多一个 fort_for_rm）
V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_2: dict[str, TerrainCommandOverrideCfg] = {
    # 小台阶（旋转版）
    "tiny_step_rot": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 上坡地形（低难度）
    "slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 下坡地形（低难度）
    "inv_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 台阶-坡道（低难度）
    "stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 台阶-坡道（中难度）
    "stair_slope_for_rm_mid": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.4)],  # 机身高度范围
    ),
    # 反台阶-坡道（低难度）
    "inv_stair_slope_for_rm_low": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 平面（旋转版）
    "plane_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.20, 0.42)],  # 机身高度范围
    ),
    # 随机均匀地形
    "random_uniform_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.22, 0.42)],  # 机身高度范围
    ),
    # 堡垒地形（版本 2 新增）
    "fort_for_rm": TerrainCommandOverrideCfg(
        height_range=[(0.24, 0.4)],  # 机身高度范围
    ),
    # 台阶（旋转版）：低速前后各一段，非航向角速度 ±1
    "stair_for_rm_rot": TerrainCommandOverrideCfg(
        height_range=[(0.2, 0.42)],  # 机身高度范围
        lin_vel_x=[(0.5, 2.0), (-2.0, -0.5)],  # x 速度：低速前后各一段
        lin_vel_y=(0.0, 0.0),  # y 速度固定 0
        # ang_vel_z_heading=(-1, 1),  # 注释掉航向角速度
        ang_vel_z_non_heading=(-1., 1.),  # 非航向角速度 ±1
        reset_heading_axis_aligned_only=True,  # reset 只做轴对齐航向
        disable_predefined_reset_air=True,  # 禁用预定义空中 reset
        disable_special_mode=True,  # 禁用特殊模式
    ),
}


def _enable_v14_rough_height_offset_curriculum(terrain_gen, num_levels: int) -> None:
    """Enable row-bucket scaling for cliff height offsets in V14 rough terrain."""
    # 中文：为 V14 崎岖地形启用"按行（难度等级）缩放高度偏移"的课程学习

    # 地形行数 = 难度等级数（每一行对应一个难度）
    terrain_gen.num_rows = int(num_levels)
    # 遍历所有子地形，为带高度偏移的子地形开启按难度缩放
    for sub_cfg in terrain_gen.sub_terrains.values():
        # 读取该子地形的高度偏移范围（没有则跳过）
        height_offset_range = getattr(sub_cfg, "height_offset_range", None)
        if height_offset_range is None:
            continue
        # 若高度偏移全为 0，则无需课程缩放
        if not any(abs(float(value)) > 1.0e-9 for value in height_offset_range):
            continue
        # 仅当该子地形支持"按难度缩放高度偏移"时，写入相关属性
        if hasattr(sub_cfg, "height_offset_curriculum_scale_by_difficulty"):
            sub_cfg.height_offset_curriculum_scale_by_difficulty = True  # 开启按难度缩放
            sub_cfg.height_offset_curriculum_num_levels = int(num_levels)  # 难度等级数


def _filter_v14_terrain_command_overrides(
    overrides: dict[str, TerrainCommandOverrideCfg],
    terrain_gen,
) -> dict[str, TerrainCommandOverrideCfg]:
    """Keep only command overrides matching the active V14 rough terrain keys."""
    # 中文：只保留"当前地形生成器确实包含的子地形名"对应的命令覆盖项

    # 取出地形生成器里实际存在的子地形键集合
    terrain_keys = set(getattr(terrain_gen, "sub_terrains", {}).keys())
    # 过滤：只留下键在 terrain_keys 里的覆盖项，并深拷贝避免外部引用被改
    return copy.deepcopy({key: value for key, value in overrides.items() if key in terrain_keys})


def _apply_v14_rough_runtime_cfg(cfg) -> None:
    """Apply the shared V14 rough-terrain configuration to a composed env cfg."""
    # 中文：把一个"组合好的 env cfg"统一改造为 V14 崎岖地形运行时配置

    # 高度定义方式统一：不用腿长当高度、也不使用绝对高度（即用相对地面高度）
    cfg.use_leg_length_as_height = False
    cfg.use_absolute_height = False
    # 崎岖地形需要启用状态机（空中/楼梯等）
    cfg.enable_state_machines = True
    # 深拷贝空中状态机配置后强制开启
    cfg.airborne_state_machine_cfg = copy.deepcopy(cfg.airborne_state_machine_cfg)
    cfg.airborne_state_machine_cfg["enabled"] = True
    # 深拷贝轮子前向扫描配置后强制开启
    cfg.wheel_forward_scan_cfg = copy.deepcopy(cfg.wheel_forward_scan_cfg)
    cfg.wheel_forward_scan_cfg["enabled"] = True
    # 关闭常规 curriculum（崎岖地形用高度偏移课程代替）
    cfg.curriculum = None

    # 开启"终止时长"逻辑
    cfg.termination_duration_enabled = True

    # 以默认 cfg 为底，叠加用户在 cfg 上提供的 rough_height_offset_curriculum_cfg
    rough_height_offset_curriculum_cfg = dict(V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG)
    raw_rough_height_offset_curriculum_cfg = getattr(cfg, "rough_height_offset_curriculum_cfg", None)
    if raw_rough_height_offset_curriculum_cfg is not None:
        rough_height_offset_curriculum_cfg.update(raw_rough_height_offset_curriculum_cfg)
    cfg.rough_height_offset_curriculum_cfg = rough_height_offset_curriculum_cfg

    # 依次确定"崎岖地形生成器"来源：显式字段 → terrain.terrain_generator → 默认 RM_ROUGH_TERRAINS_CFG
    raw_rough_terrain_gen = getattr(cfg, "rough_terrain_generator_cfg", None)
    if raw_rough_terrain_gen is None:
        raw_rough_terrain = getattr(cfg, "terrain", None)
        raw_rough_terrain_gen = getattr(raw_rough_terrain, "terrain_generator", None)
    if raw_rough_terrain_gen is None:
        raw_rough_terrain_gen = mdp.RM_ROUGH_TERRAINS_CFG

    # 深拷贝地形生成器并开启 curriculum
    rough_terrain_gen = copy.deepcopy(raw_rough_terrain_gen)
    rough_terrain_gen.curriculum = True
    # 若启用高度偏移课程，则按等级数启用缩放
    if cfg.rough_height_offset_curriculum_cfg["enabled"]:
        _enable_v14_rough_height_offset_curriculum(
            rough_terrain_gen, cfg.rough_height_offset_curriculum_cfg["num_levels"]
        )

    # 选择命令覆盖表（优先用 cfg 提供的，否则用默认 V14_ROUGH_TERRAIN_COMMAND_OVERRIDES）
    raw_terrain_command_overrides = getattr(
        cfg, "rough_terrain_command_overrides_cfg", V14_ROUGH_TERRAIN_COMMAND_OVERRIDES
    )
    # 过滤掉当前地形生成器里不存在的条目
    cfg.terrain_command_overrides = _filter_v14_terrain_command_overrides(
        raw_terrain_command_overrides, rough_terrain_gen
    )
    # 决定地形导入器：优先用 cfg 提供的 rough_terrain_importer_cfg，其次用 cfg.terrain
    raw_terrain_importer = copy.deepcopy(getattr(cfg, "rough_terrain_importer_cfg", None))
    if raw_terrain_importer is None:
        raw_terrain_importer = copy.deepcopy(getattr(cfg, "terrain", None))
    if raw_terrain_importer is not None:
        # 覆写为 generator 地形，并挂上我们的崎岖地形生成器
        cfg.terrain = raw_terrain_importer
        cfg.terrain.terrain_type = "generator"
        cfg.terrain.terrain_generator = rough_terrain_gen
    else:
        # 都没有时，新建一个默认的 generator 地形导入器（含乘性摩擦材质）
        cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",  # 地面 prim 路径
            terrain_type="generator",  # 地形类型：程序化生成
            collision_group=-1,  # 碰撞组
            terrain_generator=rough_terrain_gen,  # 使用上面的崎岖地形生成器
            # max_init_terrain_level=0,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",  # 摩擦按乘性组合
                restitution_combine_mode="multiply",  # 恢复系数按乘性组合
                static_friction=1.0,  # 静摩擦系数
                dynamic_friction=1.0,  # 动摩擦系数
                restitution=0.0,  # 恢复系数（完全不弹）
            ),
            debug_vis=False,  # 关闭地形可视化
        )

    # 启用机身高度扫描器
    _enable_v14_body_height_scanner(cfg)
    # 只要任一"扫描相关"功能开启，就启用左右轮高度扫描器；否则关闭
    if bool(cfg.airborne_state_machine_cfg.get("enabled", False)) or bool(
        getattr(cfg, "wheel_forward_scan_cfg", {}).get("enabled", False)
    ) or bool(
        getattr(cfg, "stair_state_machine_cfg", {}).get("enabled", False)
    ):
        _enable_v14_wheel_height_scanners(cfg)
    else:
        _disable_v14_wheel_height_scanners(cfg)

    # 调试用数值诊断开关（默认关）
    cfg.debug_value_diagnosis = False
    cfg.debug_value_diagnosis_interval = 200  # 诊断间隔
    cfg.debug_value_diagnosis_topk = 5  # 打印 top-k


def _make_v14_body_height_scanner_cfg() -> RayCasterCfg:
    """Create the default V14 body raycaster cfg when a parent config disabled it."""
    # 中文：当父 cfg 关闭了机身高度扫描器时，创建一个 V14 默认的扫描器配置

    return RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",  # 挂在机身 base_link 上
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),  # 从机身上方 5m 往下投射
        ray_alignment="yaw",  # 射线只跟随偏航（保持水平）
        pattern_cfg=patterns.GridPatternCfg(
            resolution=V14_BODY_HEIGHT_SCANNER_RESOLUTION,  # 网格分辨率
            size=V14_BODY_HEIGHT_SCANNER_GRID_SIZE,  # 网格尺寸
        ),
        debug_vis=False,  # 关闭可视化
        mesh_prim_paths=["/World/ground"],  # 只对地面做 raycast
    )


def _enable_v14_task_flag_obs(cfg) -> None:
    """Append a reserved task flag slot next to command observations."""
    # 中文：在观测中追加一个"任务标志"占位维度，并同步更新观测维度/裁剪/空间定义
    # 判断 observation_space 是否为 dict（是则要按子键分别更新维度）
    obs_space_is_dict = isinstance(getattr(cfg, "observation_space", None), dict)
    cfg.task_flag_obs_enabled = True  # 标记已启用任务标志观测
    cfg.task_flag_obs_dim = V14_TASK_FLAG_DIM  # 任务标志维度
    cfg.num_single_obs = V14_BASE_POLICY_OBS_DIM + V14_TASK_FLAG_DIM  # policy 单帧维度 = 28+1
    cfg.num_single_privileged_obs = (
        getattr(cfg, "num_single_privileged_obs", V14_BASE_PRIVILEGED_OBS_DIM) + V14_TASK_FLAG_DIM
    )  # 特权单帧维度 = 32+1
    cfg.state_space = cfg.num_privileged_obs_hist * cfg.num_single_privileged_obs  # 状态空间 = 特权历史帧数 × 特权单帧维

    # 在观测裁剪配置中加入 task_flag 项（范围 [-10, 10]）
    obs_clip_cfg = dict(getattr(cfg, "obs_input_clip_cfg", {}))
    obs_clip_cfg["task_flag"] = [-10.0, 10.0]
    cfg.obs_input_clip_cfg = obs_clip_cfg

    if obs_space_is_dict:
        # dict 结构：分别更新 policy/policy_hist/critic/prev_critic/critic_hist
        cfg.observation_space = dict(cfg.observation_space)
        cfg.observation_space["policy"] = cfg.num_single_obs * cfg.num_obs_hist  # policy = 单帧×历史帧
        cfg.observation_space["policy_hist"] = cfg.num_single_obs * cfg.num_obs_hist  # 同 policy
        cfg.observation_space["critic"] = cfg.num_single_privileged_obs  # critic = 特权单帧
        cfg.observation_space["prev_critic"] = cfg.num_single_privileged_obs  # 上一帧 critic
        cfg.observation_space["critic_hist"] = cfg.num_single_privileged_obs * cfg.num_privileged_obs_hist  # critic 历史
    else:
        # 非 dict：按是否帧堆叠决定观测空间大小
        if bool(getattr(cfg, "use_frame_stack", False)):
            cfg.observation_space = cfg.num_obs_hist * cfg.num_single_obs  # 帧堆叠：历史帧×单帧
        else:
            cfg.observation_space = cfg.num_single_obs  # 不堆叠：仅单帧


def _set_v14_observation_dims(cfg, policy_dim: int, privileged_dim: int) -> None:
    """Set policy and privileged observation dimensions for isolated experiments."""
    # 中文：为隔离实验直接设置 policy/privileged 观测维度，并同步派生量
    cfg.num_single_obs = int(policy_dim)  # policy 单帧维度
    cfg.num_single_privileged_obs = int(privileged_dim)  # 特权单帧维度
    cfg.state_space = cfg.num_privileged_obs_hist * cfg.num_single_privileged_obs  # 状态空间

    if isinstance(getattr(cfg, "observation_space", None), dict):
        # dict 结构：分别更新各子键维度
        cfg.observation_space = dict(cfg.observation_space)
        cfg.observation_space["policy"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["policy_hist"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["prev_critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["critic_hist"] = cfg.num_single_privileged_obs * cfg.num_privileged_obs_hist
    else:
        # 非 dict：按是否帧堆叠决定
        cfg.observation_space = cfg.num_obs_hist * cfg.num_single_obs if bool(
            getattr(cfg, "use_frame_stack", False)
        ) else cfg.num_single_obs


def _enable_v14_body_height_scanner(cfg) -> None:
    """Apply V14-specific body height scanner settings."""
    # 中文：应用 V14 专用的机身高度扫描器设置（若不存在则先创建）
    if getattr(cfg, "height_scanner", None) is None:
        cfg.height_scanner = _make_v14_body_height_scanner_cfg()  # 缺失时用默认配置创建
    # 统一覆盖网格图案（分辨率 + 尺寸）
    cfg.height_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=V14_BODY_HEIGHT_SCANNER_RESOLUTION,  # 分辨率
        size=V14_BODY_HEIGHT_SCANNER_GRID_SIZE,  # 尺寸
    )
    cfg.height_scanner.ray_alignment = "yaw"  # 射线只跟随偏航
    cfg.height_scanner.debug_vis = False  # 关闭可视化
    cfg.height_scanner.mesh_prim_paths = ["/World/ground"]  # 只对地面 raycast


def _enable_v14_wheel_height_scanners(cfg) -> None:
    """Attach dedicated raycasters to the left/right wheel centers."""
    # 中文：给左/右轮心各挂一个专用 raycastor（用于轮子离地/触地判断）
    base_scanner = copy.deepcopy(cfg.height_scanner)  # 以机身扫描器为模板
    base_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=V14_WHEEL_HEIGHT_SCANNER_RESOLUTION,  # 轮子扫描分辨率
        size=V14_WHEEL_HEIGHT_SCANNER_GRID_SIZE,  # 轮子扫描尺寸
    )
    base_scanner.ray_alignment = "yaw"  # 射线只跟随偏航
    base_scanner.debug_vis = False  # 关闭可视化

    right_scanner = copy.deepcopy(base_scanner)  # 复制一份给右轮
    right_scanner.prim_path = "/World/envs/env_.*/Robot/right_wheel_link"  # 挂在右轮

    left_scanner = copy.deepcopy(base_scanner)  # 复制一份给左轮
    left_scanner.prim_path = "/World/envs/env_.*/Robot/left_wheel_link"  # 挂在左轮

    cfg.right_wheel_height_scanner = right_scanner  # 写入 cfg
    cfg.left_wheel_height_scanner = left_scanner  # 写入 cfg


def _disable_v14_wheel_height_scanners(cfg) -> None:
    """Disable dedicated wheel raycasters when scan-based helpers are unused."""
    # 中文：当不使用基于扫描的功能时，关闭左右轮专用 raycastor
    cfg.right_wheel_height_scanner = None  # 右轮扫描器置空
    cfg.left_wheel_height_scanner = None  # 左轮扫描器置空


def _apply_v14_flat_runtime_optimizations(cfg) -> None:
    """Keep V14 plane tasks on the lightweight absolute-height path.

    Flat V14 does not need terrain-aware state machines or any raycast-based
    helpers. We force the plane variant onto absolute-height observations so
    scan sensors are not created accidentally through inherited defaults.
    """
    # 中文：让 V14 平地任务走"轻量绝对高度"路径，避免误建 raycast 传感器
    terrain_cfg = getattr(cfg, "terrain", None)  # 取地形配置
    if getattr(terrain_cfg, "terrain_type", None) != "plane":
        return  # 非平地任务直接跳过（本优化只针对 plane）

    cfg.use_leg_length_as_height = False  # 不用腿长当高度
    cfg.use_absolute_height = True  # 使用绝对高度
    cfg.enable_state_machines = False  # 平地不需要状态机
    cfg.airborne_state_machine_cfg = copy.deepcopy(cfg.airborne_state_machine_cfg)  # 深拷贝后再改
    cfg.airborne_state_machine_cfg["enabled"] = False  # 关闭空中状态机
    cfg.wheel_forward_scan_cfg = copy.deepcopy(cfg.wheel_forward_scan_cfg)  # 深拷贝后再改
    cfg.wheel_forward_scan_cfg["enabled"] = False  # 关闭轮子前向扫描
    cfg.play_height_scanner_debug_vis = False  # 关闭演示时的高度扫描可视化


# V14 崎岖地形 NP3O Barlow PlusPriv 任务的奖励权重表（有序字典，键=奖励项名，值=权重）
# 仅 WheelbipeV14RoughNP3OBarlowPlusPrivEnvCfg 使用；作为崎岖地形默认策略目标
V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_REWARDS = OrderedDict(
    # REWARD MAP V14_ROUGH_NP3O_BARLOW_PLUS_PRIV:
    # - Only used by WheelbipeV14RoughNP3OBarlowPlusPrivEnvCfg.
    # - Uses the selected balance-gated setting as the default rough policy target.
    termination=-500.0,  # 终止惩罚
    # leg_joint_acc=-2.5e-4,  # 备选（注释）
    leg_joint_acc=-2.5e-6,  # 腿关节加速度惩罚
    # joint_vel=-5.0e-5,  # 备选（注释）
    leg_joint_vel=-1.0e-3,  # 腿关节速度惩罚
    joint_torque=-2.0e-5,  # 关节力矩惩罚
    rear2_rear1_joint_pos_limits=-3.0,  # 后腿关节位置越界惩罚
    rear2_rear1_joint_pos_limits_torque=-5.0,  # 后腿位置越界时的力矩惩罚
    rear2_rear1_joint_pos_limits_vel=-5.0,  # 后腿位置越界时的速度惩罚
    wheel_power=-1.0e-3,  # 轮功率惩罚
    wheel_air_spin=-1.0e-2,  # 轮子空转惩罚
    lin_vel_z=-0.2,  # 机体系 z 速度（上下颠）惩罚
    ang_vel_xy=-0.002,  # 俯仰/翻滚角速度惩罚
    action_smoothness_leg=-0.05,  # 腿动作二阶平滑惩罚
    action_rate=-0.05,  # 动作一阶变化率惩罚
    action_smoothness_wheel=-0.01,  # 轮动作二阶平滑惩罚
    flat_orientation_y=-2.0,  # y 方向姿态（roll）惩罚
    flat_orientation_y_v=-2.0,  # y 方向姿态（速度自适应版）惩罚
    flat_orientation_x=-0.2,  # x 方向姿态（pitch）惩罚
    flat_orientation_x_v=-0.5,  # x 方向姿态（速度自适应版）惩罚
    track_lin_vel_xy=5.0,  # x 线速度跟踪奖励（指数型）
    track_lin_vel_xy_tight=2.5,  # x 线速度跟踪奖励（严格型）
    track_lin_vel_xy_square=-5.0,  # x 线速度跟踪误差（平方型）惩罚
    track_ang_vel_z=1.0,  # z 角速度跟踪奖励
    track_ang_vel_z_square=-0.5,  # z 角速度跟踪误差（平方型）惩罚
    stand_still_lin_vel=-2.0,  # 站立时线速度惩罚
    stand_still=-0.0,  # 站立惩罚（权重 0，占位）
    track_height_exp=0.4,  # 高度跟踪奖励（指数型）
    track_height_exp_soft=0.2,  # 高度跟踪奖励（软版）
    # track_height_exp_tight=2.0,  # 备选（注释）
    track_height_exp_tight=1.0,  # 高度跟踪奖励（严格版）
    # track_height_exp_both_wheels_contact=0.2,  # 备选（注释）
    track_height_exp_both_wheels_contact=0.0,  # 双轮触地时高度跟踪奖励（权重 0）
    # no_fork_exp=-5.0,  # 备选（注释）
    # no_fork_z_exp=-5.0,  # 备选（注释）
    no_fork_exp=-1.0,  # 左右轮 x 方向劈叉惩罚（指数软版）
    no_fork_z_exp=-0.0,  # 左右轮 z 方向劈叉惩罚（权重 0）
    undesired_contact=-5.0,  # 非期望接触惩罚
    # wheel_motor_z_axis_align_exp=1.0,  # 备选（注释）
    # wheel_motor_z_axis_align_exp_tight=1.0,  # 备选（注释）
)


# V14 崎岖地形 NP3O Barlow PlusPriv 任务的 cost（约束）配置
V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_COSTS = {
    # COST MAP V14_ROUGH_NP3O_BARLOW_PLUS_PRIV:
    # - Only used by WheelbipeV14RoughNP3OBarlowPlusPrivEnvCfg.
    # - Cost channel order is defined by Wheelbipe25v3Env._get_np3o_costs:
    #   body_tilt, body_height, body_ang_vel_xy, torque_limit, joint_velocity_limit.
    "num_costs": V14_NP3O_COST_DIM,  # cost 通道数 = 5
    "np3o_cost_d_values": [0.0, 0.0, 0.0, 0.0, 0.0],  # 各 cost 的目标阈值 d
    "np3o_cost_k_initial": [1.0, 1.0, 1.0, 0.5, 0.5],  # 各 cost 的初始惩罚系数 k
    "np3o_tilt_limit_deg": 30.0,  # 机身倾斜角上限（度）
    "np3o_body_height_min": 0.18,  # 机身高度下限
    "np3o_body_height_max": 0.42,  # 机身高度上限
    "np3o_ang_vel_xy_limit": 8.0,  # 俯仰/翻滚角速度上限
    "np3o_torque_limit": 30.0,  # 力矩上限
    "np3o_joint_velocity_limit": 80.0,  # 关节速度上限
    "np3o_cost_clip": 100.0,  # cost 裁剪上限
}


def _apply_v14_rough_np3o_barlow_plus_priv_cost_cfg(cfg) -> None:
    """Apply the isolated NP3O cost config for the V14 rough NP3O Barlow PlusPriv task."""
    # 中文：把上面的 NP3O cost 配置逐项写入 cfg（深拷贝，避免共享引用）

    for key, value in V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_COSTS.items():
        setattr(cfg, key, copy.deepcopy(value))  # 逐键设置到 cfg 上


# 模块公开接口：只有列在 __all__ 里的名字才会被 `from ... import *` 导出
__all__ = (
    # ---- 观测维度相关常量 ----
    "V14_TASK_FLAG_DIM",
    "V14_BASE_POLICY_OBS_DIM",
    "V14_BASE_PRIVILEGED_OBS_DIM",
    "V14_PRIV_OBS_PLUS_EXTRA_DIM",
    "V14_PRIV_OBS_PLUS_PRIVILEGED_DIM",
    "V14_JOINT_SINCOS_EXTRA_DIM",
    "V14_JOINT_SINCOS_POLICY_OBS_DIM",
    "V14_JOINT_SINCOS_PRIVILEGED_OBS_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_ACTION_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_EXTRA_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_POLICY_OBS_DIM",
    "V14_JOINT_SINCOS_ACT_SINCOS_PRIVILEGED_OBS_DIM",
    # ---- 课程学习默认配置 ----
    "V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG",
    # ---- 各算法估计状态/历史帧数常量 ----
    "V14_DREAMWAQ_ESTIMATED_STATE_DIM",
    "V14_DREAMWAQ_POLICY_HIST",
    "V14_SEMA_ESTIMATED_STATE_DIM",
    "V14_SEMA_POLICY_HIST",
    "V14_RESIDUAL_ESTIMATED_STATE_DIM",
    "V14_RESIDUAL_POLICY_HIST",
    "V14_HIM_ESTIMATED_STATE_DIM",
    "V14_HIM_POLICY_HIST",
    "V14_NP3O_POLICY_HIST",
    "V14_NP3O_EST_DIM",
    "V14_NP3O_COST_DIM",
    "V14_NP3O_ON_CONSTRAINT_DIM",
    "V14_NP3O_ON_CONSTRAINT_EXTRA_PRIV_DIM",
    # ---- 观测裁剪/缩放配置 ----
    "V14_BASIC_OBS_CLIP",
    "V14_BASIC_OBS_SCALE",
    "V14_EXTRA_OBS_CLIP",
    "V14_EXTRA_OBS_SCALE",
    # ---- 扫描器参数 ----
    "V14_BODY_HEIGHT_SCANNER_GRID_SIZE",
    "V14_BODY_HEIGHT_SCANNER_RESOLUTION",
    "V14_WHEEL_HEIGHT_SCANNER_GRID_SIZE",
    "V14_WHEEL_HEIGHT_SCANNER_RESOLUTION",
    # ---- 关节/连杆命名与运动学常量 ----
    "V14_ORDERED_LEG_JOINT_NAMES",
    "V14_ORDERED_LEG_BODY_NAMES",
    "V14_LINKS_LENGTH",
    "_V14_ALPHA0_DEG",
    "_V14_ALPHA2_DEG",
    "V14_ALPHA_OFFSET",
    # ---- reset / 观测缩放 / 地形命令覆盖表 ----
    "V14_PREDEFINED_RESET_GROUND",
    "V14_LEGACY_OBS_INPUT_SCALE_CFG",
    "V14_ROUGH_TERRAIN_COMMAND_OVERRIDES",
    # "V14_ROTATION_TERRAIN_COMMAND_OVERRIDES",  # 已弃用（拆分成了 _1/_2）
    "V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1",
    "V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_2",
    # ---- 辅助函数 ----
    "_enable_v14_rough_height_offset_curriculum",
    "_filter_v14_terrain_command_overrides",
    "_apply_v14_rough_runtime_cfg",
    "_enable_v14_task_flag_obs",
    "_set_v14_observation_dims",
    "_enable_v14_body_height_scanner",
    "_enable_v14_wheel_height_scanners",
    "_disable_v14_wheel_height_scanners",
    "_apply_v14_flat_runtime_optimizations",
    # ---- NP3O Barlow PlusPriv 奖励/代价配置 ----
    "V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_REWARDS",
    "V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_COSTS",
    "_apply_v14_rough_np3o_barlow_plus_priv_cost_cfg",
)
