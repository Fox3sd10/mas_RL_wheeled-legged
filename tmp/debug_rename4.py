from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf
import shutil

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
TMP = "/home/brown/wheeled-legged_RL/tmp/dbg4.usd"
shutil.copy2(SRC, TMP)

stage = Usd.Stage.Open(TMP)
layer = stage.GetRootLayer()
dp = stage.GetDefaultPrim().GetName()
old = f"/{dp}/Left_front_link"; new = f"/{dp}/left_front1_link"

# Try 1: static Rename constructor, no callbacks via direct apply
e = Sdf.NamespaceEdit.Rename(old, "left_front1_link")
print(">>> Rename edit:", e, "currentPath:", e.currentPath, "newPath:", e.newPath)
bne = Sdf.BatchNamespaceEdit()
bne.Add(e)
print(">>> bne.edits:", list(bne.edits) if bne.edits else None)

# Attempt: ScheduleEditBatch then ApplyRootNamespaceEdit? Try Process with None callbacks
for name, fn in [
    ("Process(None,None,True)", lambda: bne.Process(None, None, True)),
]:
    try:
        r = fn()
        print(f">>> {name} -> {r}")
    except Exception as ex:
        print(f">>> {name} EXC: {ex}")

# Check
print(">>> root new:", layer.GetPrimAtPath(new) is not None)
print(">>> root old:", layer.GetPrimAtPath(old) is not None)
layer.Save()
layer.Export(TMP + ".usda")
txt = open(TMP + ".usda").read()
print(">>> usda has new:", "left_front1_link" in txt, "old:", "Left_front_link" in txt)
app.close()
