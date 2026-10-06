"""临时脚本：查 base_link 的世界 z 包围盒（用来确定 '车体下端触地' 时的高度）。

用法：python scripts/_check_base_height.py
结果打印到终端 + 写入 scripts/_base_height_result.txt
"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom  # noqa: E402

USD = "./source/agent_world/agent_world/assets/usd_files/cod_balance/cod_balance.usd"

stage = Usd.Stage.Open(USD)
lines = []
lines.append("=== 含 base_link 的 prim ===")

base_prim = None
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if "base_link" in p.lower():
        lines.append("FOUND: " + p)
        if base_prim is None and p.lower().endswith("base_link"):
            base_prim = prim

lines.append("")
lines.append("=== base_link 世界包围盒 ===")
if base_prim is not None:
    tm = Usd.TimeCode.Default()
    bb = UsdGeom.Imageable(base_prim).ComputeWorldBound(tm, UsdGeom.Tokens.default_)
    rng = bb.ComputeAlignedRange()
    if not rng.IsEmpty():
        mn, mx = rng.GetMin(), rng.GetMax()
        lines.append(f"z_min = {mn[2]:.4f}   z_max = {mx[2]:.4f}")
        lines.append(f"x [{mn[0]:.4f}, {mx[0]:.4f}]   y [{mn[1]:.4f}, {mx[1]:.4f}]")
        lines.append("")
        lines.append(">>> 若 base_link 原点在世界 z=0，则'车体下端触地'时 root_pos_z = "
                     f"{mn[2]:.4f} m")
    else:
        lines.append("(包围盒为空)")
    lines.append("")
    lines.append("=== base_link 子节点世界 z 范围 ===")
    for c in base_prim.GetChildren():
        cb = UsdGeom.Imageable(c).ComputeWorldBound(tm, UsdGeom.Tokens.default_)
        if cb:
            r = cb.ComputeAlignedRange()
            if not r.IsEmpty():
                lines.append(f"  {c.GetPath()}: z=[{r.GetMin()[2]:.4f}, {r.GetMax()[2]:.4f}]")
else:
    lines.append("没找到 base_link")

text = "\n".join(lines)
print(text)
with open("scripts/_base_height_result.txt", "w") as f:
    f.write(text + "\n")

simulation_app.close()
