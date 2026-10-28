Constrained mixture G&R models and studies; run from `cmm/`: `python -m studies <study> [key=value ...]`.

| path | content |
|---|---|
| `hcmm/` | HCMM: `core.py` (explicit, default), `core_implicit.py`, `growth.py`, `production.py`, `removal.py` |
| `fcmm/` | full CMM with cohort histories (reference, not in the FE code) |
| `setups.py` | Maes & Famaey models A–E, cases U/S/F, builders (`build_hcmm`, `VARIANTS`) |
| `wall.py` | generic wall composition (constituent recipes → HCMM states per Gauss point) |
| `studies/` | one file per group; list in `studies/__init__.py`; shared: `common.py`, `model_e.py`; Weiss in `studies/weiss/` |
| `analysis/` | tortuosity index, videos of .pvd series |
| `results/` | outputs per group/study; `result.md` (what each folder is), `modeling_history.md` (findings) |
| `jobs/` | Slurm scripts (VSC-5) |

- Do not open `compiled/` (JAX cache), `scratch/`, or result folders with .vtu series; read the study's `.txt` log instead.
- Append modeling findings (problem, cause, fix) to `results/modeling_history.md`.
- Generic code in `wall.py`, study-specific walls and parameters in the study.
