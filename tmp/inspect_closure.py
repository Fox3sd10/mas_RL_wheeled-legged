from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

USD = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"
stage = Usd.Stage.Open(USD)

NAMES = ["left_f3r1_joint", "left_f4r2_joint", "right_f3r1_joint", "right_f4r2_joint",
         "left_front3_joint", "left_front4_joint"]

for prim in stage.Traverse():
    if prim.GetName() in NAMES:
        print(f">>> {prim.GetName()}  ({prim.GetTypeName()})")
        # joint 类型属性
        for a in prim.GetAttributes():
            n = a.GetName()
            if any(k in n for k in ["limit", "axis", "localPos", "localRot", "break", "token"]):
                print(f"      {n} = {a.Get()}")
        # 是否挂了 DriveAPI
        has_drive = prim.HasAPI(UsdPhysics.DriveAPI)
        print(f"      HasDriveAPI = {has_drive}")
        if has_drive:
            for tok in ["linear", "angular"]:
                d = UsdPhysics.DriveAPI.Get(prim, tok)
                if d:
                    print(f"        drive[{tok}] type={d.GetTypeAttr().Get()} "
                          f"stiff={d.GetStiffnessAttr().Get()} damp={d.GetDampingAttr().Get()} "
                          f"maxForce={d.GetMaxForceAttr().Get()} target={d.GetTargetPositionAttr().Get()}")
        print()
app.close()
