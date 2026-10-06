from pxr import Usd, UsdPhysics
USD = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(USD)
print(">>> ALL JOINTS:")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
        n0 = b0[0].name if b0 else "world"; n1 = b1[0].name if b1 else "world"
        print(f"  {prim.GetName():30s} | {n0} -> {n1}")
print(">>> ALL LINKS:")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  {prim.GetName()}")
