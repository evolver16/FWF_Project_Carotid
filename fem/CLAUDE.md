Total-Lagrangian FE, independent of the G&R models. Design: `docs/fem.html`, usage: `docs/fem_usage.html`.

| file | content |
|---|---|
| `core.py` | `System` (assembly, Newton + continuation + PTC, `equilibrium` with custom adjoint, `run`, `arc_length`, `stability`), `BC`, `LinearSolver` (PARDISO / SuperLU), `wall_basis`, `transfer`, `anderson` |
| `elements.py` | hex8, tet10, tet4, faces quad4 / tri6 / tri3, Abaqus face numbering |
| `mesh.py` | generators (box, quarter_cylinder, tube, half_tube, bent_tube, carotid_bifurcation), .inp / .msh I/O, VTK |
| `abaqus_interface/` | `umat.py` (UMAT / UHYPER as models), `inp.py` (`fem-abaqus` runner), `vs_abaqus.py` (`fem-vs-abaqus`), `odb_export.py` (Abaqus Python) |
| `paraview_plugin.py` | ParaView Python plugin: `FEM Model` reader (.inp / .msh sets, .inp BCs over the steps), `FEM Sets` filter (sets on a result .pvd) |
| `tensor3.py`, `pytree.py` | 3x3 helpers, pytree registration of state classes |

- Model interface per Gauss point: `sigma_solver(state, F) -> (cauchy, aux)`, `commit(state, F, aux)`, optional `J_target`, `wss`.
- Formulations: `standard`, `fbar`, `hybrid` (Q1/P0, P2/P0), `hybrid_p1` (tet10 P2/P1). Arterial G&R: `hybrid` / `hybrid_p1`.
- After changing `fem/`: run `python -m verification.fem_analytic`, `fem_consistency`, `fem_umat` from the repo root.
