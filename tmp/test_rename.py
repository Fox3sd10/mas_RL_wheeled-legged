from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf
import shutil

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
TMP = "/home/brown/wheeled-legged_RL/tmp/test_rename_out.usd"
shutil.copy2(SRC, TMP)

stage = Usd.Stage.Open(TMP)
layer = stage.GetRootLayer()
dp = stage.GetDefaultPrim().GetName()
old = f"/{dp}/Left_front_link"
new = f"/{dp}/left_front1_link"
print(">>> old exists:", layer.GetPrimAtPath(old) is not None)
print(">>> new exists:", layer.GetPrimAtPath(new) is not None)

edits = Sdf.BatchNamespaceEdit()
edits.Add(Sdf.NamespaceEdit(old, new))
print(">>> num edits:", edits.edits and len(edits.edits))

def has_obj(path):
    r = layer.GetPrimAtPath(path) is not None
    return r
def can_edit(edit):
    return True

res = edits.Process(has_obj, can_edit, True)
print(">>> Process result:", res)
print(">>> after: old exists:", layer.GetPrimAtPath(old) is not None,
      "new exists:", layer.GetPrimAtPath(new) is not None)
layer.Save()
stage2 = Usd.Stage.Open(TMP)
print(">>> reopen new exists:", stage2.GetPrimAtPath(new) is not None)
app.close()
