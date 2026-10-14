"""ParaView: Warp By Vector with the displacement u, colored by von Mises, time steps of the .pvd.
Macro (one click on the selected source): Macros > Import new macro... > this file.
At startup: PVD=<file.pvd> paraview --script=docs/paraview/warp_u.py"""

import os

from paraview.simple import *

src = OpenDataFile(os.environ["PVD"]) if os.environ.get("PVD") else GetActiveSource()
warp = WarpByVector(Input=src)
warp.Vectors = ["POINTS", "u"]
view = GetActiveViewOrCreate("RenderView")
Hide(src, view)
disp = Show(warp, view)
ColorBy(disp, ("POINTS", "von_mises"))
disp.RescaleTransferFunctionToDataRange(True, False)
disp.SetScalarBarVisibility(view, True)
GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
view.ResetCamera()
Render()
