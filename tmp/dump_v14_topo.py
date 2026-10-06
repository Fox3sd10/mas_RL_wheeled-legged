import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics
usd = "/home/brown/wheeled-legged_RL/source/agent_world/agent_world/assets/usd_files/wheelbipeV14_2_1/wheelbipeV14_2.usd"
stage = Usd.Stage.Open(usd)
print("=== JOINTS body0 -> body1 ===")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        n0 = b0[0].name if b0 else "?"
        n1 = b1[0].name if b1 else "?"
        print(f"{prim.GetName():30s} {prim.GetTypeName():22s} {n0} -> {n1}")
app.close()
