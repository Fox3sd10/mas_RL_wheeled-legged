"""只读：dump 你的 COD USD 拓扑 + 关节类型。不修改任何文件。"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

usd = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
print(">>> OPEN:", usd)
stage = Usd.Stage.Open(usd)
print(">>> defaultPrim:", stage.GetDefaultPrim().GetName() if stage.GetDefaultPrim() else None)

print("\n>>> ARTICULATION-ROOT / RIGID-BODY links:")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"   [ArticulationRoot] {prim.GetPath()}")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"   [RigidBody] {prim.GetName():40s} {prim.GetPath()}")

print("\n>>> ALL JOINTS (name | type | body0 -> body1):")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        n0 = b0[0].name if b0 else "<world>"
        n1 = b1[0].name if b1 else "<world>"
        print(f"   {prim.GetName():40s} {prim.GetTypeName():24s} {n0} -> {n1}")

print("\n>>> DONE")
app.close()
