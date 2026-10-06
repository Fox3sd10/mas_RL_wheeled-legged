from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(SRC)
# find the f3r1 joint prim (old build has Left_Closure_Joint1)
p = stage.GetPrimAtPath("/COD_2026_Balance_2_0/joints/Left_Closure_Joint1")
print(">>> prim valid:", bool(p))
for prop in p.GetProperties():
    print(">>> prop:", prop.GetName(), type(prop), "IsA(Relationship):", prop.IsA(Usd.Relationship))
    if "body" in prop.GetName().lower():
        try:
            rel = Usd.Relationship(prop)
            print("   targets:", [str(t) for t in rel.GetTargets()])
        except Exception as e:
            print("   cast err:", e)
app.close()
