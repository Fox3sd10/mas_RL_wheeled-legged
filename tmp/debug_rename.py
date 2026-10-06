from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf
import shutil

SRC = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
TMP = "/home/brown/wheeled-legged_RL/tmp/dbg.usd"
shutil.copy2(SRC, TMP)

stage = Usd.Stage.Open(TMP)
layer = stage.GetRootLayer()
dp = stage.GetDefaultPrim().GetName()
old = f"/{dp}/Left_front_link"; new = f"/{dp}/left_front1_link"
print(">>> before old:", layer.GetPrimAtPath(old) is not None, "new:", layer.GetPrimAtPath(new) is not None)

edits = Sdf.BatchNamespaceEdit()
edits.Add(Sdf.NamespaceEdit(old, new))
def h(p): return layer.GetPrimAtPath(p) is not None
def c(e): return True
ok, pending = edits.Process(h, c, True)
print(">>> Process ok:", ok, "pending:", pending)
print(">>> in-memory after: old:", layer.GetPrimAtPath(old) is not None, "new:", layer.GetPrimAtPath(new) is not None)
layer.Save()
print(">>> saved")

# reopen via new stage
stage2 = Usd.Stage.Open(TMP)
l2 = stage2.GetRootLayer()
print(">>> reopen old:", l2.GetPrimAtPath(old) is not None, "new:", l2.GetPrimAtPath(new) is not None)
# close stage then reopen
Usd.Stage.Close(stage); Usd.Stage.Close(stage2)
stage3 = Usd.Stage.Open(TMP)
l3 = stage3.GetRootLayer()
print(">>> after close/reopen old:", l3.GetPrimAtPath(old) is not None, "new:", l3.GetPrimAtPath(new) is not None)
print(">>> prim names sample:", [p.GetName() for p in stage3.Traverse() if "front1" in p.GetName() or "Left_front_link" in p.GetName()])
app.close()
