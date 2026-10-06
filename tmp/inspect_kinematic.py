from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
from pxr import Usd, UsdPhysics
USD = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(USD)
print(">>> RigidBodyAPI kinematicEnabled per link:")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI(prim)
        k = rb.GetKinematicEnabledAttr().Get()
        print(f"  {prim.GetName():28s} kinematicEnabled={k}")
