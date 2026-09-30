# FWF_Project_Carotid

Growth and remodeling (G&R) of the carotid artery with constrained mixture models in a
differentiable finite element code (JAX). The homogenized constrained mixture model (HCMM, Maes &
Famaey 2023) runs in the FE code; the full constrained mixture model (FCMM) serves as reference.
Gradients of any result with respect to the model parameters come from the adjoint.

## Layout

```
fem/            FE framework, independent of the G&R models
  core.py         assembly, Newton, time loop (System); hex8/tet10, standard/F-bar/hybrid elements, PARDISO/SuperLU
  mesh.py         generators (box, quarter cylinder, half tube, bent stenotic tube), Abaqus .inp / gmsh .msh, ParaView .vtu/.pvd
  elements.py     shape functions, Gauss rules;  tensor3.py  3x3 det/inv
materials/      constitutive laws, one file each (Fung, NeoHookean, NeoHookeanInc), shared by fem, hcmm and fcmm
cmm/            constrained mixture models and studies
  hcmm/           core (constituent, mixture, commit), growth (F_g), production, removal
  fcmm/           same structure, cohort histories; independent of hcmm
  setups.py       Maes & Famaey Table 1/2 models, builders
  studies.py      all studies (python studies.py <study>)
  jobs/           Slurm scripts for VSC-5
  results/        outputs, described in results/result.md
verification/   material_point.py (0D reference, Maes U/S/F cases), fem_analytic.py (FE vs closed-form solutions)
```

Model interface used by `fem.System` and `material_point` at every integration point:
`sigma, aux = model.sigma_solver(state, F)`, `state = model.commit(state, F, aux)`.
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
python studies.py fe                                   # a study; list: python studies.py -h
python studies.py turnover p_gr=0.012 k_plus=0.3 days=3000 name=p12   # key=value parameters
python -m verification.fem_analytic                    # FE vs uniaxial / equibiaxial / simple shear / dilatation
sbatch jobs/run.sh turnover p_gr=0.012                 # on the cluster (set -A in the script)
```

| Study | Content |
|---|---|
| `verify`, `maes`, `sweep`, `compare`, `fem` | material point: HCMM vs FCMM vs exact solution, Maes Fig. 2, FE readiness |
| `fe`, `cylinder`, `incompressible`, `tet10` | FE verification: patch test, Lamé, locking, tet10, adjoint vs finite differences |
| `artery`, `grad` | G&R on a quarter cylinder, gradients and parameter identification |
| `vessel` | bent stenotic vessel from an .inp mesh, ParaView output, adjoint |
| `bending`, `local_growth` | mass redistribution under bending, growth-only displacement |
| `turnover` | pressure step: elastic vs turnover part of the widening |

Results and their interpretation: [cmm/results/result.md](cmm/results/result.md).

## Extending

- Material: new file in `materials/` with `sigma(F)`, `sigma_f(sigma)` (HCMM also `F_r(F_e, J, c, rate)`), import in `materials/__init__.py`.
- Turnover law: class in `cmm/hcmm/production.py` / `removal.py` with `increment(c, sigma_f, ds, rho_tot, rho_tot_0)`.
- Growth law: class in `cmm/hcmm/growth.py` with `F_g(ratio)`.
