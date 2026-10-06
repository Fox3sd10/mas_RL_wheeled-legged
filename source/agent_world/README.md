# agent_world 模块详解

> 位置：`source/agent_world/agent_world/`
> 作者：SCUTRobotLab（Zhang Zhirui、Cui Yu）
> 定位：**仿真世界层**——把真实机器人模型、真实电机特性、自定义地形带入 Isaac Lab。

---

## 0. 这个包是干什么的

`agent_world` 是整个 wheeled-legged（轮腿复合）机器人 RL 项目的**"世界/资产层"**。
它回答三个问题：

1. **机器人长什么样** → `assets/`（USD 模型 + `ArticulationCfg`）
2. **机器人的关节由什么驱动** → `actuators/`（自定义电机模型，sim-to-real 核心）
3. **机器人走在什么地面上** → `terrains/`（自定义地形生成器）

它**不包含**任何强化学习算法或训练循环，那些在兄弟包 `agent_tasks/`（环境/MDP）和 `agent_rl/`（策略/训练器）里。

### 依赖关系图

```
          agent_rl (PPO/NP3O 等算法)
                │  依赖
                ▼
          agent_tasks (Isaac Lab 环境、MDP 奖励/终止/观测)
                │  依赖 robot_cfg 与 terrain_cfg
                ▼
    ┌───────────────────────────────────────┐
    │            agent_world                 │
    │  assets/       actuators/   terrains/  │
    │  (机器人)      (电机)       (地形)      │
    └───────────────────────────────────────┘
                │  依赖
                ▼
    isaaclab (Isaac Lab 2.x) + Isaac Sim 4.5 / 5.1
```

### 包结构

```
source/agent_world/
├── setup.py                     # 安装脚本（包名 agent_world，Python 3.10~3.11）
├── pyproject.toml               # 构建后端
└── agent_world/
    ├── __init__.py              # 暴露 AssetPath / RootPath 路径常量
    ├── assets/                  # 【机器人模型定义】
    │   ├── wheelbipe_V13.py             # V13 系（含 NS 无弹簧版）
    │   ├── wheelbipe_V14.py             # V14 系（guide1、无云台）
    │   ├── wheelbipe_V14_2.py           # V14.2 系（当前主力 ★）
    │   ├── wheelbipe25_v3.py            # 25v3 系（含 DelayPD/DCMotor 变体）
    │   ├── wheelbipe25_v3_reduce_spring.py
    │   └── usd_files/                    # 实际的 USD 网格/物理文件
    │       └── wheelbipeV14_2_1/
    ├── actuators/               # 【自定义电机执行器模型】
    │   ├── m3508_actuator.py            # 大疆 M3508 转矩-转速包络 ★
    │   ├── diff_vel_actuator.py        # 有限差分测速 PD
    │   ├── learned_velocity_actuator.py # 神经网络学出来的速度控制器
    │   └── learned_velocity_actuator_cfg.py
    └── terrains/                # 【自定义地形生成器】
        ├── __init__.py
        └── height_field.py             # 高度场/网格地形函数集
```

---

## 1. `__init__.py` —— 路径常量

这是整个包的入口，只做一件事：**定义资产路径**，让其他文件无论从哪个工作目录启动都能找到 USD。

```python
import os

RootPath  = os.path.dirname(os.path.abspath(__file__))
AssetPath = os.path.join(RootPath, "assets")          # ← 机器人 cfg 里用 f"{AssetPath}/..."

ROBOT_UTILS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "./")
)
```

**为什么重要**：所有机器人配置里都写 `usd_path=f"{AssetPath}/usd_files/..."`，
用**绝对路径**避免了"脚本在不同目录下运行时找不到 USD"的问题。这是 Isaac Lab 项目的常见坑。

---

## 2. `assets/` —— 机器人模型定义

### 2.1 这是一个什么样的机器人：Wheelbipe（轮腿复合）

**Wheelbipe = Wheel（轮）+ Bipe**，RM（RoboMaster）步兵机器人，每条腿是**四连杆/平行四边形腿式机构**，末端装一个**驱动轮**，再加**二自由度云台**。

它既能像腿式机器人一样蹲起/跨障，又能像轮式机器人一样高速滑行。

### 2.2 USD 模型分层（以 V14.2 为例）

```
usd_files/wheelbipeV14_2_1/
├── wheelbipeV14_2.usd              # 主入口（组合层，引用下面 4 个）
├── wheelbipeV14_2_ng.usd           # NG(No Guide) 变体：去掉 guide_link
└── configuration/
    ├── wheelbipeV14_2_base.usd     # 28MB 网格几何（所有 link 的 mesh）
    ├── wheelbipeV14_2_physics.usd  # 关节/碰撞/物理（RevoluteJoint）
    ├── wheelbipeV14_2_robot.usd    # 机器人装配定义
    └── wheelbipeV14_2_sensor.usd   # 传感器挂载点
```

主 USD 通过 `subLayers`/`references` 组合四个子层。**这是从 CAD（SolidWorks）导出的分层 USD**，不是 URDF。

### 2.3 机器人关节结构

从 `cfg_utils.py` 的 `V14_ORDERED_LEG_JOINT_NAMES` 可还原 12 个腿关节（左右对称各 6 个）：

| 关节组 | 名字 | 角色 |
|---|---|---|
| 主驱动腿关节 | `.*_rear1_joint`, `.*_front1_joint` | **主动**，达妙 DM8009 电机驱动（大腿摆动）|
| 被动腿关节 | `.*_rear2_joint`, `.*_front2/3/4_joint` | **被动**，四连杆约束从动 |
| 弹簧关节 | `.*_spring1_joint`, `.*_spring2_joint` | 腿部弹簧减震（spring2 主动阻尼）|
| 导向关节 | `.*_guide_joint` | 导向/滑块（V14 有，NG 版没有）|
| 轮关节 | `.*_wheel_joint` | 轮子驱动 |
| 云台关节 | `gimbal_yaw_joint`, `gimbal_pitch_joint` | 云台偏航/俯仰 |

**link（连杆）**：`base_link`（唯一根）、`.*_front1~4_link`、`.*_rear1/2_link`、`.*_spring1/2_link`、`.*_guide_link`、`left/right_wheel_link`、`gimbal_yaw/pitch_link`。

### 2.4 `ArticulationCfg` 的通用骨架

每个机器人配置都是一个 `ArticulationCfg`，包含四块：

```python
Robot_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(...),       # ① 加载哪个 USD + 物理属性
    init_state=ArticulationCfg.InitialStateCfg(...),  # ② 初始位姿/关节角
    actuators={...},                        # ③ 关节驱动器（最核心）
)
```

**① spawn —— 生成/物理**
```python
usd_path=f"{AssetPath}/usd_files/...",   # USD 路径
activate_contact_sensors=True,            # 打开接触传感器（算触地奖励用）
copy_from_source=True,                    # Isaac Lab 2.3.0 必需，保证多环境实例化
rigid_props=RigidBodyPropertiesCfg(       # 刚体属性
    disable_gravity=False,
    max_depenetration_velocity=1.0,
),
articulation_props=ArticulationRootPropertiesCfg(
    fix_root_link=False,                  # 根不固定（浮动基座）
    enabled_self_collisions=False,        # 关自碰撞（省算力）
    solver_position_iteration_count=12,   # 求解器迭代（V14.2 用 12/6，V13 用 8/4）
    solver_velocity_iteration_count=6,
),
```

**② init_state —— 初始状态**
```python
pos=(0.0, 0.0, 0.38),   # 初始世界坐标（V14.2 是 0.38，V13/V14 是 0.35）
joint_pos={".*_rear1_joint": 0.0, ...},   # 各关节初始角
joint_vel={".*": 0.0},
```

**③ actuators —— 驱动器**（见下节）

### 2.5 驱动器分组（以 `wheelbipe_V14_2.py` 为例）

V14.2 的驱动器是 V14 系列里最完整的（含弹簧+云台）：

| 组名 | 关节 | 类型 | stiffness / damping | armature | 作用 |
|---|---|---|---|---|---|
| `legs_act` | `.*_rear1/front1` | `IdealPDActuatorCfg` | 60 / 2.0 | **DM8009_ARMATURE** | 主动大腿电机 |
| `legs_inact` | `.*_rear2`, `.*_front2/3/4`, `.*_spring1`, `.*_guide` | `IdealPDActuatorCfg` | 0 / 0.01 | 0.0001 | 被动关节 |
| `wheel` | `.*_wheel_joint` | `IdealPDActuatorCfg` | 0 / 0.2 | 0 | 轮驱电机 |
| `spring` | `.*_spring2_joint` | `IdealPDActuatorCfg` | 0 / **50** | 0.0001 | 弹簧阻尼 |
| `gimbal_yaw` | `gimbal_yaw_joint` | `IdealPDActuatorCfg` | 0 / 0.5 | 0.0001 | 云台偏航（速度控制）|
| `gimbal_pitch` | `gimbal_pitch_joint` | `IdealPDActuatorCfg` | 20 / 0.5 | 0.0001 | 云台俯仰（位置控制）|

**关键细节：达妙电机惯量**
```python
DM8009_ARMATURE = 1.95e-04 * 9.0 * 9.0      # = 0.0158 kg·m²
#                 转子惯量   × 减速比²（9:1）
```
这个 `armature`（折算到关节的电机转子惯量）**极影响仿真稳定性**。
代码里留了很多注释掉的备选值（`*0.1`、`0`），说明这是调试重点。

### 2.6 三种模型变体（V14.2 为例）

```python
Wheelbipe_V14_2_CFG = ArticulationCfg(...)          # 默认，带 gimbal + guide

Wheelbipe_V14_2_NG_CFG = Wheelbipe_V14_2_CFG.copy() # No Guide 变体
Wheelbipe_V14_2_NG_CFG.spawn.usd_path = ".../wheelbipeV14_2_ng.usd"
Wheelbipe_V14_2_NG_CFG.actuators["legs_inact"].joint_names_expr = [...]  # 去掉 guide_joint

Wheelbipe_V14_2_M3508_CFG = Wheelbipe_V14_2_CFG.copy()   # 轮子换成真实 M3508 电机
Wheelbipe_V14_2_M3508_CFG.actuators["wheel"] = M3508ActuatorCfg(
    curve_path="experiments/.../m3508_..._curve_dense.csv",
    gear_ratio=268.0 / 17.0, output_stall_torque=5.5, ...
)
```

### 2.7 各版本资产文件对比总览

| 文件 | 机器人 | 特点 |
|---|---|---|
| `wheelbipe25_v3.py` | 25v3 | 最老，`ImplicitActuator` 为主；含 `_IdealPD`/`_DelayPD`/`_guide` 变体 |
| `wheelbipe25_v3_reduce_spring.py` | 25v3 减弹簧 | 简化弹簧的消融实验版 |
| `wheelbipe_V13.py` | V13 | 加了 `DCMotorCfg` 轮子变体、`NS`（No Spring）版本 |
| `wheelbipe_V14.py` | V14 | guide1 版、无云台版；引入 `DM8009_ARMATURE` 和 M3508 |
| `wheelbipe_V14_2.py` | **V14.2 ★** | 当前主力；6 组驱动器齐全，弹簧+云台，最完善 |

**版本演进主线**：
```
25v3 (ImplicitActuator)
  └─ V13 (引入 DCMotor/NS 变体，精细化弹簧)
       └─ V14 (自定义 M3508 执行器，DM8009 惯量)
            └─ V14.2 (完善云台+弹簧，当前训练基座)
```

---

## 3. `actuators/` —— 自定义执行器（sim-to-real 核心）

这是 `agent_world` 最有技术含量的部分。Isaac Lab 自带 `ImplicitActuator` / `IdealPDActuator` / `DCMotorCfg`，
但项目**自己写了三个**，因为内置的满足不了 sim-to-real 需求。

### 3.1 `m3508_actuator.py` —— 大疆 M3508 转矩-转速包络 ★★★

**要解决的问题**：真实 M3508 电机力矩随转速升高而衰减（反电动势 + 电流环限制），
而简单 `effort_limit=常数` 会让策略学到"高速也能满力矩"的假行为 → 真车失效。

**核心机制**：把**实测的转矩-转速曲线**注入 PD 执行器。

```
CSV (电机轴实测曲线, 505 点)
   │  ① × gear_ratio(268/17=15.76) → 轮端力矩
   │  ② ÷ gear_ratio, RPM→rad/s    → 轮端转速
   ▼
轮端曲线: speed ∈ [25.03, 63.37] rad/s(实测段)
   │  ③ 低速段用 tail 拟合 + smoothstep 外推到 5.5 N·m 堵转
   ▼
完整包络: [0, 拟合堵转速] → 5.5 N·m 恒定
          [拟合堵转速, 25.03] → smoothstep 过渡
          [25.03, 63.37]      → CSV 线性插值
          [>63.37]            → 0 N·m
   ▼
每个仿真步: max_torque = searchsorted(|joint_vel|) 查表 + 线性插值
   ▼
_clip_effort(): applied = clip(PD输出, ±envelope(|vel|))
```

**三个关键方法**：
- `_load_torque_speed_curve()`：读 CSV，做单位换算，调用低速外推
- `_prepend_low_speed_extension()`：用最低速 64 点线性拟合趋势，反推达到 5.5 N·m 的速度，用 **smoothstep（S 形平滑曲线）** 生成 128 个合成点，保证与实测段 **C¹ 连续**（避免力矩跳变）
- `_lookup_torque_limit()`：`torch.searchsorted` 二分 + 线性插值 + 边界钳制

**配置参数**：

| 参数 | 默认 | 含义 |
|---|---|---|
| `curve_path` | MISSING | CSV 路径 |
| `gear_ratio` | 268/17 = 15.76 | 自定义减速箱比例 |
| `gearbox_efficiency` | 1.0 | 减速箱效率（首版不扣摩擦）|
| `output_stall_torque` | 5.5 N·m | 堵转力矩上限 |
| `tail_fit_points` | 64 | 拟合低速趋势用的点数 |
| `tail_extension_points` | 128 | 合成低速点数 |

**数值验证（已实跑确认）**：

| 电机轴 | → 轮端 |
|---|---|
| 9540.35 RPM | 63.3734 rad/s |
| 3768.63 RPM | 25.0338 rad/s |
| 0.262014 N·m | 4.1306 N·m |

**训练验证**：实验 005 用 4096 环境训练 5000 轮，reward 108.73，episode length 816.52 → **可训练**。

### 3.2 `diff_vel_actuator.py` —— 有限差分测速 PD

**要解决的问题**：真机上的关节速度是**用位置差分估计的**（编码器读数差分），不是仿真里的真值。
要让 policy 看到/控制与真机一致的测速器。

**核心机制**：`IdealPDActuator` 的 PD 律不变，但把速度反馈替换成 `(q_t - q_{t-1}) / dt`。

```python
def compute(self, control_action, joint_pos, joint_vel):
    diff_vel = self._estimate_joint_vel(joint_pos, joint_vel)
    return super().compute(control_action, joint_pos, diff_vel)   # 用差分速度替代真值

def _estimate_joint_vel(...):
    delta_pos = joint_pos - prev_joint_pos
    if wrap_to_pi: delta_pos = atan2(sin(Δ), cos(Δ))   # 角度环绕处理
    reset_mask = |Δ| > threshold                        # 检测 reset 跳变
    diff_vel = Δ / diff_dt
    if reset_mask: diff_vel = fallback_vel              # reset 时回退到仿真速度
    diff_vel = EMA滤波(diff_vel)                        # 可选平滑
```

**配置**：`diff_dt=0.005`、`wrap_to_pi`、`velocity_filter_alpha`、`reset_position_jump_threshold`。

**用途**：sim-to-real 时让仿真噪声/延迟与真机测速器匹配。

### 3.3 `learned_velocity_actuator.py` —— 神经网络速度控制器 ★

**要解决的问题**：最激进的 sim-to-real——**用真机数据训练一个神经网络**，
直接学出"速度指令 → 力矩"的映射，彻底替代解析式 PD。

**核心机制**：加载 TorchScript 模型，用**滚动历史缓冲区**喂入时序特征。

```python
def compute(self, control_action, joint_pos, joint_vel):
    vel_cmd = control_action.joint_velocities
    # 构造特征 [速度指令, 当前速度, kd系数] → [B, J, 3]
    feat = stack([vel_cmd, joint_vel, kd_tensor], dim=-1)
    # 推入循环缓冲区（H 步历史）
    self._buffer[:, :, ptr, :] = feat
    # 归一化 + reshape [B*J, H, 3]
    n = (buf_seq - mean) / std
    # 跑模型
    effort = self._model(n).reshape(B, J)
    effort = clamp(effort, ±effort_limit)     # 力矩钳制
    effort = EMA平滑(effort)                   # 可选
    control_action.joint_efforts = effort
```

**特点**：
- **状态无关**：不经过 `IdealPDActuator` 的 PD 律，直接输出力矩
- **归一化统计**：从 `norm_stats.json` 读 mean/std（与训练一致）
- **惰性分配**：缓冲区在第一次 `compute()` 时才创建（因为 num_envs/joints 运行时才知道）
- **路径解析**：同 M3508，支持绝对/相对/仓库根三级查找

**配置**（`learned_velocity_actuator_cfg.py`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `model_path` | MISSING | TorchScript 模型路径 |
| `norm_stats_path` | MISSING | 归一化统计 JSON |
| `history_len` | 10 | 历史步数 H |
| `kd` | 0.175 | 训练时的硬件阻尼系数（作为常量输入）|
| `effort_limit` | 5.0 N·m | 力矩上限 |
| `effort_smoothing` | 0.0 | EMA 平滑系数 |

### 3.4 执行器选型对比

| 执行器 | 原理 | sim-to-real 价值 | 复杂度 |
|---|---|---|---|
| `ImplicitActuator`（内置）| PhysX 内置 PD | 低 | 最低 |
| `IdealPDActuator`（内置）| 显式 PD | 低 | 低 |
| **`M3508Actuator`** | PD + 实测转矩-转速包络 | **高**（力矩衰减）| 中 |
| **`DiffVelPDActuator`** | PD + 差分测速 | **中**（测速器匹配）| 中 |
| **`LearnedVelocityActuator`** | 神经网络直出力矩 | **最高**（学真机动力学）| 高 |

---

## 4. `terrains/` —— 自定义地形生成器

### 4.1 整体

`height_field.py` 定义了一批 **Isaac Lab 地形函数 + 对应 `*Cfg`**。
Isaac Lab 地形有两种表示：
- **HeightField（高度场）**：2D 数组，逐像素高度，用 `@height_field_to_mesh` 装饰后自动转网格。
- **Mesh（网格）**：直接用 trimesh 拼盒子，更精确但更重。

所有地形都用 **difficulty（难度 0~1）** 驱动参数插值——ISaac Lab 课程学习的基础。

### 4.2 地形清单

| 地形 | 类型 | 用途 |
|---|---|---|
| `FourQuadrantTerrainCfg` | 组合 | 一块地形切成 4 个象限，各自独立配置 |
| `HfCustomNpyTerrainCfg` | HeightField | 从 NPY 文件加载自定义高度场（可导入真实地形数据）|
| `HfCustomGridBarsTerrainCfg` | HeightField | 横竖格栅（hash/grid）障碍 |
| `MeshCustomGridBarsTerrainCfg` | Mesh | 同上，用盒子网格，水平杆在竖直杆处切段避免重叠 |
| `MeshCustomSplitGridBarsTerrainCfg` | Mesh | 一半横杆一半竖杆，**完全避免交叉**（对应 RM 的"梅花桩"）|
| `HfCliffInvertedPyramidStairsTerrainCfg` | HeightField | **反金字塔阶梯**（四周悬崖，中心下沉）|
| `HfCustomRaisedInvertedPyramidSlopedTerrainCfg` | HeightField | 抬高的反金字塔斜坡 |
| `HfCustomDirectionalWaveTerrainCfg` | HeightField | 单轴正弦波地形 |
| `HfCustomTruncatedSlopedTerrainCfg` | HeightField | 单轴斜坡（可多个、可随机）|

### 4.3 通用设计模式

**① "固定值 or 难度范围"二选一**：
```python
def _resolve_fixed_or_difficulty_value(fixed_value, value_range, difficulty, name):
    if fixed_value is not None:
        return float(fixed_value)              # 固定值
    lower, upper = value_range
    return lower + difficulty * (upper - lower) # 按难度插值
```
绝大多数参数（步高、杆宽、角度）都这样写，让地形随课程难度变难。

**② 高度场函数签名**（被 `@height_field_to_mesh` 装饰）：
```python
@height_field_to_mesh
def some_terrain(difficulty, cfg) -> np.ndarray:
    ...
    return np.rint(hf_raw).astype(np.int16)   # 返回以 vertical_scale 为单位的整数高度
```

**③ 网格函数签名**（返回 list[Trimesh] + origin）：
```python
def mesh_terrain(difficulty, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    meshes_list = [...]
    origin = np.asarray([size_x/2, size_y/2, z])
    return meshes_list, origin
```

### 4.4 亮点实现

**Mesh 格栅避免重叠面**（`mesh_grid_bars_terrain`）：
> 竖直杆做全长盒子，水平杆在竖直杆跨度处**切段**，
> 让交叉处只被一个盒子占据，而不是两个重叠盒子。
> 这样俯视形状等价于 max 合成，但**避免了重复共面接触面**（否则仿真会抖动）。

**反金字塔阶梯 + 高度课程**（`HfCliffInvertedPyramidStairsTerrainCfg`）：
配合 `_resolve_height_offset_curriculum`，用"难度桶"（row-bucket）缩放高度偏移，
让课程学习按行渐进提升悬崖高度。

**对称杆中心计算**（`_symmetric_bar_centers`）：
保证杆与杆之间有**相等的间隙**、两侧有**对称的边距**——地形对称，机器人左/右行为才对称。

### 4.5 MDP 地形的下游

`agent_tasks` 里的 `mdp/terrain`（如 `RM_ROUGH_TERRAINS_CFG`、`TerrainCommandOverrideCfg`）
会消费这些地形 cfg，把它们组装成完整的地形生成器，并按地形名切换命令范围。

---

## 5. 完整数据流

```
                      ┌────────────────────────────────┐
                      │  SolidWorks / CAD 模型          │
                      └───────────────┬────────────────┘
                                      │ 导出 → Isaac Sim USD 转换
                                      ▼
        usd_files/*.usd  (base/physics/robot/sensor 四层)
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
   assets/*.py               actuators/*.py            terrains/height_field.py
   (ArticulationCfg)         (电机执行器 Cfg)           (地形 Cfg)
   - spawn (USD+物理)         - M3508 曲线               - 格栅/阶梯/斜坡/波
   - init_state              - 差分测速 PD               - 难度插值
   - actuators {6 组}         - 学习型速度控制器          - 高度场/Mesh
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                    agent_tasks (Isaac Lab 环境)
                    - robot_cfg = V14_2_CFG.replace(prim_path=...)
                    - terrain_cfg = 组装后的地形
                    - MDP: 观测/奖励/终止/课程
                                      │
                                      ▼
                    agent_rl (PPO / NP3O / HIM / DreamWaQ ...)
                    - 训练循环、策略网络、checkpoint
                                      │
                                      ▼
                              预训练策略 → 真机部署
```

---

## 6. 关键概念速查表

| 概念 | 含义 |
|---|---|
| **轮腿机器人** | 腿式机构 + 末端轮，兼具跨越与滑行能力 |
| **主动/被动关节** | `front1/rear1` 有电机（主动）；`front2~4`、`rear2` 是四连杆从动 |
| **armature** | 电机转子惯量（折算到关节），影响仿真稳定性 |
| **转矩-转速曲线** | 真实电机建模，高速时力矩衰减（sim-to-real 关键）|
| **smoothstep** | S 形平滑插值，保证外推段与实测段 C¹ 连续 |
| **USD 分层** | base(几何)/physics(关节)/robot(装配)/sensor 分离，主文件组合 |
| **HeightField vs Mesh** | 高度场（数组）vs 三角网格，前者轻后者精 |
| **difficulty** | 0~1 难度值，驱动地形参数插值（课程学习基础）|
| **sim-to-real** | 用实测曲线/惯量/摩擦/测速器逼近真车 |

---

## 7. 常见问题

**Q：机器人模型在哪？**
A：`source/agent_world/agent_world/assets/usd_files/wheelbipeV14_2_1/wheelbipeV14_2.usd`

**Q：这是 URDF 吗？**
A：不是，是 **USD**（从 CAD 转的，见 `configuration/` 四层子文件）。

**Q：机器人的"肌肉"定义在哪？**
A：`assets/*.py` 的 `actuators={...}`，每个机器人 6 组驱动器。

**Q：为什么自己写执行器而不用内置的？**
A：内置的无法表达真实电机的转矩-转速衰减、差分测速器、以及数据驱动的电机模型。

**Q：M3508 的曲线数据在哪？**
A：原始 CSV 在 `experiments/v14_flat/005_3508_motor/`，
路径写在 `Wheelbipe_V14_2_M3508_CFG.actuators["wheel"].curve_path`。

**Q：机器人和地形如何被环境消费？**
A：`agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py` 里
`robot_cfg = Wheelbipe_V14_2_CFG.replace(prim_path="/World/envs/env_.*/Robot")`。

**Q：这包能独立训练吗？**
A：不能。它只提供"世界定义"，训练需要搭配 `agent_tasks` + `agent_rl`。

---

## 8. 目录一览（速查）

```
agent_world/
├── __init__.py                    路径常量 (AssetPath)
├── assets/
│   ├── wheelbipe25_v3.py          25v3 基础 + IdealPD/DelayPD/guide 变体
│   ├── wheelbipe25_v3_reduce_spring.py  减弹簧消融
│   ├── wheelbipe_V13.py           V13 + NS无弹簧 + DCMotor 变体
│   ├── wheelbipe_V14.py           V14 guide1 + 无云台 + M3508
│   ├── wheelbipe_V14_2.py   ★      V14.2 主力（6 组驱动器齐全）
│   └── usd_files/wheelbipeV14_2_1/   USD 模型
├── actuators/
│   ├── m3508_actuator.py      ★    M3508 转矩-转速包络
│   ├── diff_vel_actuator.py        差分测速 PD
│   ├── learned_velocity_actuator.py / _cfg.py   神经网络速度控制器
├── terrains/
│   ├── __init__.py
│   └── height_field.py         ★    9 种自定义地形
```
