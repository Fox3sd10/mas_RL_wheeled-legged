from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim import SimulationContext, SimulationCfg

USD = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
sim = SimulationContext(SimulationCfg(dt=0.005, device="cpu"))
sim_utils.spawn_from_usd("/World/envs/env_0/Robot", sim_utils.UsdFileCfg(usd_path=USD))
print(">>> SPAWNED")
for prim in sim.stage.Traverse():
    p = str(prim.GetPath())
    if "Robot" in p and p.count("/") <= 6:
        print(f"    {prim.GetTypeName():24s} {p}")
