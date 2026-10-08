# FWF_Project_Carotid

Growth and remodeling (G&R) of the carotid artery with constrained mixture models in a
differentiable finite element code (JAX). The homogenized constrained mixture model (HCMM, Maes &
Famaey 2023) runs in the FE code; the full constrained mixture model (FCMM) serves as reference.
Gradients of any result with respect to the model parameters come from the adjoint.

## Layout

```
fem/            FE framework, independent of the G&R models
  core.py         assembly, Newton, time loop (System); hex8/tet10, standard/F-bar/hybrid (Q1/P0, P2/P0, P2/P1 Taylor-Hood) elements, PARDISO/SuperLU;
                  buckling: arc_length (load path through limit points), stability (tangent eigenvalues),
                  pseudo-transient relaxation as automatic fallback when Newton fails (snaps in G&R steps)
  mesh.py         generators (box, quarter cylinder, half tube, bent stenotic tube), Abaqus .inp / gmsh .msh, ParaView
  elements.py     shape functions, Gauss rules
  tensor3.py      3x3 det, inverse, polar rotation;  pytree.py  registration of model state classes
materials/      constitutive laws, one file each (Fung, NeoHookean, NeoHookeanInc), shared by fem, hcmm and fcmm
cmm/            constrained mixture models and studies
  hcmm/           core (constituent, mixture, sigma_solver, commit), growth (F_g), production, removal
  fcmm/           same structure, cohort histories; independent of hcmm
  setups.py       Maes & Famaey Table 1/2 models, constituent specs, builders
  studies/        python -m studies <study>; groups material, fe_gr, artery, vessel (table below)
  jobs/           Slurm scripts for VSC-5
  results/        outputs, described in results/result.md
verification/   python -m verification.<module>; outputs in verification/results/
  material_point.py  0D reference for the Maes U/S/F cases
  fem_analytic.py    FE vs homogeneous closed-form solutions, patch test
  fem_tube.py        thick-walled tube vs Lame / finite-strain solutions, element formulations, hex8 vs tet10
  mesh_io.py         .inp / .msh round trips
  buckling.py        Euler column, shallow arch snap-through: arc length, stability, relaxation
  fem_mms.py         manufactured solutions, convergence orders of all element formulations
  fem_consistency.py tangent, Newton order, follower pressure, invariances, free growth, adjoint
  cook.py            Cook's membrane locking benchmark
```

Model interface used by `fem.System` and `material_point` at every integration point:
`sigma, aux = model.sigma_solver(state, F)`, `state = model.commit(state, F, aux)`, optional `J_target(state)`.
The model is a module (`hcmm`, `fcmm`), the state its mixture per Gauss point.
`fem.System(mesh)` without a model uses the state as a plain material (no G&R).

## Install

Python 3.12, from the repository root:

```bash
python -m venv ~/venvs/cmm && source ~/venvs/cmm/bin/activate
pip install -r cmm/requirements.txt
pip install -e .                     # makes fem, materials, verification importable
```

On VSC-5 load `python/3.12.8-gcc-12.2.0-4y5tbpr` first; rerun `pip install -e .` after pulls that add packages.

## Run

```bash
cd cmm
python -m studies pressure_step                              # a study; list: python -m studies -h
python -m studies pressure_step p_gr=0.012 k_plus=0.3 days=3000 name=p12   # key=value parameters
sbatch jobs/run.sh pressure_step p_gr=0.012                  # on the cluster (set -A in the script)
cd .. && python -m verification.fem_tube                     # FE verification, from the repository root
```

Outputs: `cmm/results/<group>/<study>/`.

| Group | Study | Content |
|---|---|---|
| material | `exact_solution` | HCMM / FCMM vs the exact constrained mixture solution at constant F |
| material | `maes_benchmark` | Maes & Famaey models A–E, cases U/S/F (Fig. 2) |
| material | `parameter_sweep` | HCMM–FCMM gap over prestretch, gain and step size |
| material | `case_comparison` | one model / case, HCMM vs FCMM over time (plot) |
| material | `fe_readiness` | batching, objectivity, tangent and gradients of the models for the FE code |
| fe_gr | `fe_vs_material_point` | HCMM patch test, single (hybrid) elements vs the material point, adjoint |
| fe_gr | `element_formulations` | artery G&R with standard / F-bar / hybrid / tet10, tet10 adjoint with inclined supports |
| fe_gr | `solution_verification` | artery G&R: mesh and time-step convergence, GCI |
| artery | `artery_gr` | G&R of a quarter cylinder (Maes & Famaey Fig. 4) |
| artery | `artery_gradients` | adjoint gradients vs finite differences, parameter identification, cost |
| artery | `pressure_step` | pressure step held constant: elastic vs turnover part of the widening |
| artery | `flow_step` | flow step at constant pressure, WSS stimulus from the deformed lumen (fluid-coupling template) |
| artery | `pressure_buckling` | pressure buckling of a G&R state with G&R frozen: critical pressure, post-buckling path |
| vessel | `stenotic_vessel` | bent stenotic vessel from an .inp mesh, Laplace wall basis, ParaView output, adjoint |
| vessel | `bending_redistribution` | mass redistribution in a bent tube (end rotation) |
| vessel | `setpoint_patch` | lowered collagen set point in a patch: growth-only displacement |

| Verification | Content |
|---|---|
| `fem_analytic` | uniaxial / equibiaxial (hybrid), simple shear, dilatation, patch test; hex8 and tet10 |
| `fem_tube` | Lamé convergence, nonlinear refinement, standard / F-bar / hybrid, finite-strain incompressible tube, tet10 |
| `mesh_io` | Abaqus .inp, gmsh .msh 4.1 / 2.2 round trips |
| `buckling` | Euler column (stability), shallow arch (arc length vs displacement control, relaxation) |
| `fem_mms` | manufactured solutions: L2 / H1 / pressure orders for hex8, tet10, standard / F-bar / hybrid (incl. J_target ≠ 1) |
| `fem_consistency` | tangent = AD of the residual, quadratic Newton, follower load = −p dV/dx, slip / rotation / renumbering / solver invariance, free growth, adjoint vs FD |
| `cook` | Cook's membrane: standard locks, F-bar / hybrid converge |

G&R solution verification (mesh and time-step GCI of the artery): `cd cmm && python -m studies solution_verification`.

Results and their interpretation: [cmm/results/result.md](cmm/results/result.md).

## Extending

- Material: new file in `materials/` with `sigma(F)`, `sigma_f(sigma)` (HCMM also `F_r(F_e, J, c, rate)`), import in `materials/__init__.py`.
- Turnover law: class in `cmm/hcmm/production.py` / `removal.py` with `increment(c, sigma_f, ds, rho_tot, rho_tot_0, dtau)`,
  `dtau = tau_w / tau_w_h - 1` the wall shear stress stimulus (mixture fields `tau_w`, `tau_w_h`, default 1).
- Fluid input as boundary conditions: wall pressure `BC.p` and wall shear stress `BC.wss` per face Gauss point of the
  pressure faces (sample a fluid solution at `System.face_points(u)`, map with `fem.transfer`); `run` passes `BC.wss`
  to every wall Gauss point (nearest lumen point) via the model's `wss(state, tau_w)`; homeostatic `tau_w_h` in the state
  (example: study `flow_step`).
- Growth law: class in `cmm/hcmm/growth.py` with `F_g(ratio)`.
