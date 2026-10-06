from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics, Sdf

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(SRC)
p = stage.GetPrimAtPath("/COD_2026_Balance_2_0/joints/Left_Closure_Joint1")
print(">>> prim valid:", bool(p))
for prop in p.GetProperties():
    tn = prop.GetTypeName()
    print(">>> prop:", prop.GetName(), "typeName:", tn)
    if tn == "rel":
        rel = Usd.Relationship(prop)
        print("    targets:", [str(t) for t in rel.GetTargets()])
app.close()
