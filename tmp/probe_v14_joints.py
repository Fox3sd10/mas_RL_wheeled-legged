from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from agent_world.assets.wheelbipe_V14_2 import Wheelbipe_V14_2_CFG

cfg = Wheelbipe_V14_2_CFG.copy()
cfg.spawn.usd_path = cfg.spawn.usd_path  # 保持不变
cfg.actuators = {}
cfg.init_state.joint_pos = {}
cfg.init_state.joint_vel = {}

sim = SimulationContext(sim_utils.SimulationCfg(dt=1/200.0, device="cuda:0"))
ground = sim_utils.CuboidCfg(size=(100,100,0.1),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg())
ground.func("/World/ground", ground, translation=(0,0,-0.05))

robot = Articulation(cfg.replace(prim_path="/World/Robot"))
sim.reset()
print(">>> V14 ARTICULATION joints:", robot.joint_names)
print(">>> num_joints:", robot.num_joints)
# 检查闭链关节是否在里面
for jn in ["left_f3r1_joint","left_f4r2_joint","left_spring2_joint","left_front_guide_joint"]:
    print(f"   {jn} in articulation? {jn in robot.joint_names}")
app.close()
