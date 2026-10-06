from pxr import Usd
usd_path = "/home/brown/wheeled-legged_RL/source/agent_world/agent_world/assets/usd_files/wheelbipeV14_2_1/configuration/wheelbipeV14_2_robot.usd"
stage = Usd.Stage.Open(usd_path)
print("=== ROBOT PRIM TREE ===")
for prim in stage.Traverse():
    print(f"{prim.GetTypeName():25s} | {prim.GetPath()}")
