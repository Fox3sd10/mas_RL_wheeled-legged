from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf
import shutil

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
TMP = "/home/brown/wheeled-legged_RL/tmp/dbg_apply.usd"
shutil.copy2(SRC, TMP)

stage = Usd.Stage.Open(TMP)
layer = stage.GetRootLayer()
dp = stage.GetDefaultPrim().GetName()
old = f"/{dp}/Left_front_link"; new = f"/{dp}/left_front1_link"

bne = Sdf.BatchNamespaceEdit()
bne.Add(Sdf.NamespaceEdit.Rename(old, "left_front1_link"))
print(">>> CanApply:", layer.CanApply(bne))
r = layer.Apply(bne)
print(">>> Apply result:", r)
print(">>> root old:", layer.GetPrimAtPath(old) is not None, "new:", layer.GetPrimAtPath(new) is not None)
layer.Save()
layer.Export(TMP + ".usda")
txt = open(TMP + ".usda").read()
print(">>> usda new:", "left_front1_link" in txt, "old:", "Left_front_link" in txt)
app.close()
