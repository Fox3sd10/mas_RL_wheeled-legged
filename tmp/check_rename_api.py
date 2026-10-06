from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, Sdf
import omni.usd

print(">>> Sdf free funcs:", [m for m in dir(Sdf) if "ename" in m.lower() or "Namespace" in m])
print(">>> Sdf.Layer methods:", [m for m in dir(Sdf.Layer) if not m.startswith("_")])
# Check Usd.Prim methods for rename
print(">>> Usd.Prim methods w/ Name:", [m for m in dir(Usd.Prim) if "ame" in m])
# omni.usd utils
try:
    import omni.usd
    print(">>> omni.usd has:", [m for m in dir(omni.usd) if "rename" in m.lower() or "path" in m.lower()])
except Exception as e:
    print("omni.usd err", e)
app.close()
