FE verification, run from the repo root: `python -m verification.<module>`; logs in `results/<module>.txt` (read these, not the scripts, for numbers). Overview with all results: `docs/fem_verification.html`.

| module | checks |
|---|---|
| `fem_analytic` | homogeneous deformations, patch test (hex8, tet10, tet4) |
| `fem_tube` | thick-walled tube vs Lamé / finite strain, locking, hex8 vs tet10 |
| `fem_mms` | manufactured solutions, convergence orders (slow, ~6 min) |
| `fem_consistency` | tangent, Newton order, follower pressure, invariances, adjoint |
| `cook` | Cook's membrane |
| `buckling` | Euler column, shallow arch (arc length, stability, PTC) |
| `mesh_io` | .inp / .msh round trips |
| `fem_umat` | UMAT / UHYPER interface (`umat/*.for`, needs gfortran) |
| `material_point` | 0D driver used by studies (not a test itself) |
