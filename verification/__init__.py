"""Verification, run from the repository root as python -m verification.<module>:

material_point  0D reference for the Maes U/S/F cases (homogeneous F), used by the studies and the FE checks
fem_analytic    FE vs closed-form homogeneous deformations and the patch test (hex8, tet10)
fem_tube        pressurized thick-walled tube vs Lame / finite-strain solutions, element formulations, hex8 vs tet10
mesh_io         Abaqus .inp and gmsh .msh round trips
buckling        Euler column (stability), shallow arch snap-through (arc length, relaxation)
"""
