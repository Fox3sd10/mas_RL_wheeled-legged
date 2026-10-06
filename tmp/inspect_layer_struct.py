from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd
SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(SRC)
root = stage.GetRootLayer()
print(">>> root layer:", root.identifier)
print(">>> root defaultPrim:", stage.GetDefaultPrim().GetName())
dp = stage.GetDefaultPrim()
print(">>> defaultPrim children:", [c.GetName() for c in dp.GetChildren()][:20])
print(">>> root layer sublayers:", root.subLayerPaths)
print(">>> session layer:", stage.GetSessionLayer().identifier)
# Which layer defines base_link?
bl = stage.GetPrimAtPath(f"/{dp.GetName()}/Left_front_link")
print(">>> Left_front_link exists:", bool(bl))
if bl:
    specs = bl.GetPrimStack()
    print(">>> Left_front_link primstack layers:", [s.layer.identifier for s in specs])
app.close()
