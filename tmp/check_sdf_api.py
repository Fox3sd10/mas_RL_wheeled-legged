from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Sdf
print(">>> Sdf.Layer namespace methods:")
print([m for m in dir(Sdf.Layer) if "ame" in m.lower() or "Edit" in m])
print(">>> Sdf.BatchNamespaceEdit methods:", [m for m in dir(Sdf.BatchNamespaceEdit) if not m.startswith("_")])
print(">>> NamespaceEdit methods:", [m for m in dir(Sdf.NamespaceEdit) if not m.startswith("_")])
app.close()
