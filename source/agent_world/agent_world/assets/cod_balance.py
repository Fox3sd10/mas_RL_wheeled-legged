# =============================================================================
# cod_balance.py  ——  COD 平衡机器人的"资产配置文件"(ArticulationCfg)
#
# 【这个文件是干什么的？】
#   USD 文件里只有"几何 + 关节定义"（像一张图纸）。
#   这个 .py 文件告诉 Isaac Sim：
#       - 加载哪个 USD 文件
#       - 重力/求解器/自碰撞等物理参数
#       - 初始位姿（放在哪、各关节初始角度）
#       - 每个关节用什么驱动器（电机）：刚度、阻尼、力矩上限、速度上限
#
#   这套东西合起来 = ArticulationCfg，就是一个"可被训练环境 spawn 的机器人资产"。
#
# 【机器人结构速查】（从 USD dump 得到）
#   base_link（根，挂 ArticulationRoot）
#     ├─ 左腿: left_front1 ~ left_front4 （4 节）→ left_rear1, left_rear2 → left_wheel
#     └─ 右腿: right_front1 ~ right_front4（4 节）→ right_rear1, right_rear2 → right_wheel
#   闭环约束关节(左右各2): *_f3r1_joint, *_f4r2_joint
#
#   主动关节(有电机, 6 个):  *.front1_joint, *.rear1_joint, *.wheel_joint
#   被动关节(无电机, 8 个):  *.front2/front3/front4_joint, *.rear2_joint
#   闭环关节(约束,   4 个):  *.f3r1_joint, *.f4r2_joint
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
# IdealPDActuatorCfg: 理想 PD 驱动器（最常用、最好理解）
#   给定期望位置/速度，内部用  tau = stiffness*(q_des - q) + damping*(qd_des - qd) 算力矩
#   再裁剪到 effort_limit 内。适合学习理解。后续可换 ImplicitActuatorCfg（更稳定）。
from isaaclab.actuators import IdealPDActuatorCfg

# AssetPath 指向 agent_world/agent_world/assets 目录
from agent_world import AssetPath


# =============================================================================
# 一、spawn 部分：加载 USD + 设置物理属性
# =============================================================================
COD_BALANCE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # ① 模型文件路径（我们把 USD 放在 assets/usd_files/cod_balance/ 下）
        usd_path=f"{AssetPath}/usd_files/cod_balance/cod_balance.usd",

        # ② 对每个刚体统一覆盖物理属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # ★ 关键！你的 base_link 在 USD 里被标记为 kinematic（运动学），
            #   这会导致 articulation 无法正确建立（训练时报错/不收敛）。
            #   这里强制把它变成普通动力学刚体（关闭 kinematic）。
            kinematic_enabled=False,
            disable_gravity=False,          # 开启重力（要它掉下来）
            linear_damping=0.0,             # 线性阻尼
            angular_damping=0.0,            # 角阻尼
            max_linear_velocity=1000.0,     # 最大线速度（防止数值爆炸）
            max_angular_velocity=1000.0,    # 最大角速度
            max_depenetration_velocity=1.0, # 最大穿透恢复速度
        ),

        # ③ 对整个 articulation（多刚体系统）的设置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,            # 根不固定（可自由移动/摔倒）
            enabled_self_collisions=False,  # 关闭自碰撞（腿部关节相连，开了易抖）
            solver_position_iteration_count=12,  # 位置求解迭代（越大越精确越慢）
            solver_velocity_iteration_count=6,   # 速度求解迭代
        ),
    ),

    # =====================================================================
    # 二、初始状态：spawn 时机器人放在哪、各关节初值
    # =====================================================================
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),   # 初始位置 (x, y, z)，z=0.35m 稍高于地面自由落体
        joint_pos={
            # 主动关节初始角（弧度），0 = 模型默认零位
            ".*_front1_joint": 0.0,
            ".*_rear1_joint": 0.0,
            ".*_wheel_joint": 0.0,
            # 被动关节初始角
            ".*_front2_joint": 0.0,
            ".*_front3_joint": 0.0,
            ".*_front4_joint": 0.0,
            ".*_rear2_joint": 0.0,
            # ★ 注意：不要在这里写 f3r1/f4r2（闭环关节）。
            #   Isaac 在建立 articulation 树时会自动丢弃形成环的关节，
            #   它们不在 robot.joint_names 里，写进来会报 "regex not matched"。
        },
        joint_vel={".*": 0.0},  # 所有关节初速度=0
    ),

    # =====================================================================
    # 三、actuators：驱动器（电机）配置
    #   每个 actuator 用 joint_names_expr 正则匹配它负责的关节。
    #   匹配顺序按 dict 先后，一个关节只应被一个 actuator 匹配到。
    # =====================================================================
    actuators={
        # ---- 主动腿关节：front1 / rear1（负责抬腿、摆腿、支撑）----
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_front1_joint",
                ".*_rear1_joint",
            ],
            stiffness={
                ".*_front1_joint": 60.0,   # P 增益：越大越"硬"，跟踪位置越紧
                ".*_rear1_joint": 60.0,
            },
            damping={
                ".*_front1_joint": 2.0,    # D 增益：越大越"黏"，抑制震荡
                ".*_rear1_joint": 2.0,
            },
            effort_limit={                # 最大输出力矩 (N·m)
                ".*_front1_joint": 40.0,
                ".*_rear1_joint": 40.0,
            },
            velocity_limit={              # 最大关节速度 (rad/s)
                ".*_front1_joint": 17.0,
                ".*_rear1_joint": 17.0,
            },
            armature=1.95e-4 * 9.0 * 9.0, # 电机转子惯量折算（可先设为小值，影响不大）
        ),

        # ---- 车轮：速度控制（stiffness=0 表示不做位置控制）----
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,                 # 轮子只做速度/力矩控制
            damping=15.0,
            effort_limit=6.0,              # 最大轮力矩
            velocity_limit=60.0,           # 最大轮速
            armature=0.0,
        ),

        # ---- 被动关节：几乎不施加力，仅保留一点点阻尼 ----
        #   注意：这里【不含】f3r1/f4r2 闭环关节，because Isaac 会把它们从
        #   articulation 树里剔除，写进来会 launch 报错。
        "legs_passive": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_rear2_joint",
            ],
            stiffness=0.0,                 # 不给位置刚度（跟随链接）
            damping=0.01,                  # 极小阻尼
            effort_limit=50.0,
            velocity_limit=300.0,
            armature=0.0001,
        ),
    },
)
