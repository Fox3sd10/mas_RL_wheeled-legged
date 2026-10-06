from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from agent_world.assets.cod_balance import COD_BALANCE_CFG

cfg = COD_BALANCE_CFG.copy()
# 清空 actuators / init_state，只为了拿到 articulation 真实 joint 列表
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
print(">>> ARTICULATION joints:", robot.joint_names)
print(">>> num_joints:", robot.num_joints)
print(">>> BODIES:", robot.body_names)
app.close()
