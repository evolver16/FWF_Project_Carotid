"""ParaView Python plugin (paraview --venv <venv with fem>, Tools > Manage Plugins > Load New, Auto Load):
    FEM Model   reader for .inp / .msh: mesh, 0/1 arrays nset_* / elset_* per named set, one block per face set;
                .inp over the increments of its steps: bc_fixed, bc_u (prescribed u), f_dead (*Cload), p (*Dsload)
    FEM Sets    filter on a result (.pvd / .vtu): the named sets of an .inp / .msh (default: <stem>.inp of the .pvd,
                written by VTKSeries(pvd, mesh)), face blocks share the result's points and point data (Warp By Vector)
"""

import copy
import pathlib
import sys

import numpy as np
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smdomain, smhint, smproperty, smproxy
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (vtkCellArray, vtkCompositeDataSet, vtkDataObject, vtkMultiBlockDataSet,
                                           vtkUnstructuredGrid)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fem import mesh as meshlib
from fem.abaqus_interface import inp as abq

FACE_VTK = {"quad4": 9, "tri6": 22, "tri3": 5}


def read(path):
    path = pathlib.Path(path)
    return meshlib.read_msh(path) if path.suffix.lower() == ".msh" else meshlib.read_inp(path)


def array(name, v):
    a = numpy_to_vtk(np.ascontiguousarray(v), deep=True)
    a.SetName(name)
    return a


def grid(points, conn, vtk_type):
    g = vtkUnstructuredGrid()
    g.SetPoints(points)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, conn.size + 1, conn.shape[1], dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn, dtype=np.int64).ravel(), deep=True))
    g.SetCells(vtk_type, cells)
    return g


def blocks(volume, m, p=None):
    """multiblock (volume with nset_* / elset_* arrays, faces/<name> sharing its points and point data), internal
    sets (_name) left out; p: face set name -> pressure (cell array p)"""
    named = lambda sets: [(k, v) for k, v in sets.items() if isinstance(k, str) and not k.startswith("_")]
    n, e = volume.GetNumberOfPoints(), volume.GetNumberOfCells()
    for name, v in named(m.nodes):
        volume.GetPointData().AddArray(array(f"nset_{name}", np.isin(np.arange(n), v).astype(np.uint8)))
    for name, v in named(m.elsets):
        volume.GetCellData().AddArray(array(f"elset_{name}", np.isin(np.arange(e), v).astype(np.uint8)))
    faces = vtkMultiBlockDataSet()
    for i, (name, f) in enumerate(named(m.faces)):
        g = grid(volume.GetPoints(), f, FACE_VTK[m.element.face.name])
        g.GetPointData().ShallowCopy(volume.GetPointData())
        if p is not None:
            g.GetCellData().AddArray(array("p", np.full(len(f), p.get(name.lower(), 0.0))))
        faces.SetBlock(i, g)
        faces.GetMetaData(i).Set(vtkCompositeDataSet.NAME(), name)
    out = vtkMultiBlockDataSet()
    for i, (name, b) in enumerate((("volume", volume), ("faces", faces))):
        out.SetBlock(i, b)
        out.GetMetaData(i).Set(vtkCompositeDataSet.NAME(), name)
    return out


@smproxy.reader(name="FEMModelReader", label="FEM Model", extensions="inp msh",
                file_description="FEM model (.inp, .msh)")
class FEMModelReader(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet")
        self.path, self.model, self.frames = None, None, []

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions=["inp", "msh"], file_description="FEM model (.inp, .msh)")
    def SetFileName(self, name):
        if name != self.path:
            self.path, self.model = name, None
            self.Modified()

    def load(self):
        """mesh and frames [(total time, step histories, step fraction, step time)] at every increment"""
        if self.model is None:
            path = pathlib.Path(self.path)
            m, frames = read(path), [(0.0, None, 0.0, 0.0)]
            if path.suffix.lower() == ".inp":
                kws = abq.keywords(path.read_text())
                amps = {o["name"].lower(): abq.amplitude((o, data)) for k, o, data in kws if k == "amplitude"}
                t0 = 0.0
                for _, period, n_inc, *hist in abq.steps(kws, abq.node_lookup(kws, m), amps):
                    hist = [copy.copy(h) for h in hist]
                    frames += [(t0 + i * period / n_inc, hist, i / n_inc, i * period / n_inc)
                               for i in range(1, n_inc + 1)]
                    t0 += period
            self.model, self.frames = m, frames

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        self.load()
        return [fr[0] for fr in self.frames] if len(self.frames) > 1 else None

    def RequestInformation(self, request, inInfo, outInfo):
        executive, info = self.GetExecutive(), outInfo.GetInformationObject(0)
        info.Remove(executive.TIME_STEPS())
        info.Remove(executive.TIME_RANGE())
        times = self.GetTimestepValues()
        if times:
            for t in times:
                info.Append(executive.TIME_STEPS(), t)
            info.Append(executive.TIME_RANGE(), times[0])
            info.Append(executive.TIME_RANGE(), times[-1])
        return 1

    def RequestData(self, request, inInfo, outInfo):
        self.load()
        m, executive, info = self.model, self.GetExecutive(), outInfo.GetInformationObject(0)
        t = info.Get(executive.UPDATE_TIME_STEP()) if info.Has(executive.UPDATE_TIME_STEP()) else 0.0
        _, hist, f, ts = [fr for fr in self.frames if fr[0] <= t + 1e-9 * max(1.0, abs(t))][-1]
        points = vtkPoints()
        points.SetData(array("Points", np.asarray(m.X, dtype=float)))
        volume = grid(points, m.conn, m.element.vtk)
        p = None
        if len(self.frames) > 1:
            fixed, u_bc, f_dead, p = np.zeros(3 * m.n_nodes, np.uint8), np.zeros(3 * m.n_nodes), np.zeros(3 * m.n_nodes), {}
            if hist is not None:
                bcs, loads, cloads = hist
                for k in bcs.now:
                    fixed[k], u_bc[k] = 1, bcs.value(k, f, ts)
                for k in cloads.now:
                    f_dead[k] += cloads.value(k, f, ts)
                p = {s: loads.value(s, f, ts) for s in loads.now}
            for name, v in (("bc_fixed", fixed), ("bc_u", u_bc), ("f_dead", f_dead)):
                volume.GetPointData().AddArray(array(name, v.reshape(-1, 3)))
        vtkMultiBlockDataSet.GetData(outInfo, 0).ShallowCopy(blocks(volume, m, p))
        return 1


@smproxy.filter(name="FEMSets", label="FEM Sets")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkUnstructuredGrid"], composite_data_supported=True)
class FEMSets(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=1, nOutputPorts=1, inputType="vtkDataObject", outputType="vtkMultiBlockDataSet")
        self.path, self.model = "", None

    @smproperty.stringvector(name="MeshFile", default_values="")
    @smdomain.filelist()
    @smhint.filechooser(extensions=["inp", "msh"], file_description="FEM model (.inp, .msh)")
    def SetMeshFile(self, name):
        if name != self.path:
            self.path, self.model = name, None
            self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        data = vtkDataObject.GetData(inInfo[0], 0)
        if data.IsA("vtkCompositeDataSet"):
            it = data.NewIterator()
            it.InitTraversal()
            data = it.GetCurrentDataObject()
        if self.model is None:
            path = self.path
            alg = self.GetInputAlgorithm(0, 0)
            while not path and alg is not None:
                if hasattr(alg, "GetFileName") and alg.GetFileName():
                    path = str(pathlib.Path(alg.GetFileName()).with_suffix(".inp"))
                alg = alg.GetInputAlgorithm(0, 0) if alg.GetNumberOfInputPorts() and \
                    alg.GetNumberOfInputConnections(0) else None
            if not path or not pathlib.Path(path).exists():
                raise RuntimeError(f"FEM Sets: no mesh file ({path or 'set MeshFile'})")
            self.model = read(path)
        if self.model.n_nodes != data.GetNumberOfPoints():
            raise RuntimeError(f"FEM Sets: {self.model.n_nodes} nodes in the mesh file, {data.GetNumberOfPoints()} "
                               f"points in the input")
        volume = vtkUnstructuredGrid()
        volume.ShallowCopy(data)
        vtkMultiBlockDataSet.GetData(outInfo, 0).ShallowCopy(blocks(volume, self.model))
        return 1
