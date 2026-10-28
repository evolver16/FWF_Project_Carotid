"""ParaView: deformed shape colored by von Mises (J_g or the first point array where there is none, e.g. the rod
series), time steps of the .pvd; Warp By Vector with the displacement u only for older series written at the
reference (newer ones have the deformed points and the reference as X_ref).
Macro (one click on the selected source): Macros > Import new macro... > this file.
At startup: PVD=<file.pvd> paraview --script=docs/paraview/warp_u.py"""

import os

from paraview.simple import *

src = OpenDataFile(os.environ["PVD"]) if os.environ.get("PVD") else GetActiveSource()
view = GetActiveViewOrCreate("RenderView")
shown = src
if "X_ref" not in src.PointData.keys():
    shown = WarpByVector(Input=src)
    shown.Vectors = ["POINTS", "u"]
    Hide(src, view)
disp = Show(shown, view)
names = list(src.PointData.keys())
ColorBy(disp, ("POINTS", next((n for n in ("von_mises", "J_g") if n in names), names[0])))
disp.RescaleTransferFunctionToDataRange(True, False)
disp.SetScalarBarVisibility(view, True)
GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
view.ResetCamera()
Render()
