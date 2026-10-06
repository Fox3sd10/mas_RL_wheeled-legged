from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(SRC)
n = 0
for prim in stage.Traverse():
    rels = prim.GetRelationships()
    if rels:
        print(">>>", prim.GetPath(), "rels:", [r.GetName() for r in rels])
        for r in rels:
            print("    ", r.GetName(), "->", [str(t) for t in r.GetTargets()])
        n += 1
        if n > 8:
            break
app.close()
