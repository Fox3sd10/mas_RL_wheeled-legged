from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Sdf
import inspect
print(">>> Process sig:", Sdf.BatchNamespaceEdit.Process.__doc__)
app.close()
