from pxr import Usd, UsdPhysics

usd_path = "/home/brown/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(usd_path)
print("=== PRIM TREE ===")
for prim in stage.Traverse():
    t = prim.GetTypeName()
    print(f"{t:25s} | {prim.GetPath()}")
