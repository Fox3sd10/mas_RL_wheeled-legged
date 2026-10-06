from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(SRC)
p = stage.GetPrimAtPath("/COD_2026_Balance_2_0/joints/Left_Closure_Joint1")
print(">>> joint prim:", p.GetPath(), "IsA Joint:", p.IsA(UsdPhysics.Joint))
print(">>> relationships:", [r.GetName() for r in p.GetRelationships()])
for r in p.GetRelationships():
    print("    ", r.GetName(), "->", [str(t) for t in r.GetTargets()])
j = UsdPhysics.Joint(p)
print(">>> GetBody0Rel:", [str(t) for t in j.GetBody0Rel().GetTargets()])
print(">>> GetBody1Rel:", [str(t) for t in j.GetBody1Rel().GetTargets()])
# Where's base_link? links at /COD_2026_Balance_2_0/Left_front_link
print(">>> Left_front_link valid:", bool(stage.GetPrimAtPath("/COD_2026_Balance_2_0/Left_front_link")))
app.close()
