from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from pxr import UsdPhysics, Usd

USD = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
sim = SimulationContext(SimulationCfg(dt=0.005, device="cpu"))
sim_utils.spawn_from_usd("/World/envs/env_0/Robot", sim_utils.UsdFileCfg(usd_path=USD))
stage = sim.stage
print(">>> CHECK base_link APIs")
bl = stage.GetPrimAtPath("/World/envs/env_0/Robot/base_link")
print("base_link type:", bl.GetTypeName())
print("  has RigidBodyAPI:", bl.HasAPI(UsdPhysics.RigidBodyAPI))
print("  has ArticulationRootAPI:", bl.HasAPI(UsdPhysics.ArticulationRootAPI))
print("  applied schemas:", bl.GetAppliedSchemas())
print("  has PhysicsMassAPI:", bl.HasAPI(UsdPhysics.MassAPI))
print(">>> children of base_link:", [c.GetName() for c in bl.GetChildren()][:5], "... total", len(list(bl.GetChildren())))
# Check each link
print(">>> ALL link APIs:")
for name in ["base_link","Left_front_link","Left_rear_link","Left_Wheel_link"]:
    p = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
    print(f"  {name:22s} RB={p.HasAPI(UsdPhysics.RigidBodyAPI)} Art={p.HasAPI(UsdPhysics.ArticulationRootAPI)} type={p.GetTypeName()}")
