from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf
import shutil

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
TMP = "/home/brown/wheeled-legged_RL/tmp/dbg3.usd"
shutil.copy2(SRC, TMP)

stage = Usd.Stage.Open(TMP)
layer = stage.GetRootLayer()
dp = stage.GetDefaultPrim().GetName()
old = f"/{dp}/Left_front_link"; new = f"/{dp}/left_front1_link"

edits = Sdf.BatchNamespaceEdit()
edits.Add(Sdf.NamespaceEdit(old, new))
def h(p): return layer.GetPrimAtPath(p) is not None
def c(e): return True
res = edits.Process(h, c, True)
print(">>> Process:", res)
layer.Save()

stage2 = Usd.Stage.Open(TMP)
print(">>> stage2.GetPrimAtPath(new):", stage2.GetPrimAtPath(new) is not None)
print(">>> stage2.root.GetPrimAtPath(new):", stage2.GetRootLayer().GetPrimAtPath(new) is not None)
print(">>> stage2.GetPrimAtPath(old):", stage2.GetPrimAtPath(old) is not None)
# export usda and check text
stage2.GetRootLayer().Export(TMP + ".usda")
txt = open(TMP + ".usda").read()
print(">>> usda has new:", "left_front1_link" in txt)
print(">>> usda has old:", "Left_front_link" in txt)
app.close()
