from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import UsdUtils, Sdf, Usd
print(">>> UsdUtils attrs:", [m for m in dir(UsdUtils) if not m.startswith("_")])
print(">>> Sdf attrs with 'Namespace':", [m for m in dir(Sdf) if "amespace" in m])
app.close()
