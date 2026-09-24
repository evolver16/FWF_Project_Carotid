"""FCMM vs HCMM studies: python studies.py maes | sweep | compare [MODEL CASE] | verify"""

import argparse
import csv
import pathlib
import time

import jax.numpy as jnp

import plotting
import setups
import solver
from setups import MODELS, TARGETS, VARIANTS, Spec, G_fiber, G_matrix, neo_hookean, fung

RESULTS = pathlib.Path(__file__).parent / "results"
DRIVEN = solver.DRIVEN_AXIS


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_variants(specs, bc, ds, n_steps, ag=None, variants=VARIANTS, x0=None):
    """{variant: solver.run output} for the same specs and BC."""
    return {var: solver.run(build(specs, ds, ag), module, bc, n_steps=n_steps,
                            x0=jnp.ones(bc.n_unknowns) if x0 is None else x0)
            for var, (build, module) in variants.items()}


# ---------- Maes & Famaey (2023) Table 1/2, Fig. 2 ----------

def maes(ds=10.0, n_steps=100):
    out = {}
    for model in MODELS:
        for case in TARGETS:
            t0 = time.time()
            specs = setups.maes_specs(model, case)
            res = run_variants(specs, setups.bc_for(model, case), ds, n_steps, MODELS[model]["ag"])
            for var, r in res.items():
                out[(model, case, var)] = r
            print(f"  {model} {case}  ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for model in MODELS:
        for case in TARGETS:
            f = out[(model, case, "FCMM")]
            lf, sf = f["lam_driven"], f["sigma_driven"]
            for var in ("HCMM dep", "HCMM init"):
                lh, sh = out[(model, case, var)]["lam_driven"], out[(model, case, var)]["sigma_driven"]
                rows.append(dict(
                    model=model, case=case, variant=var,
                    lam_FCMM=float(lf[-1]), lam_HCMM=float(lh[-1]),
                    sigma_FCMM=float(sf[-1]), sigma_HCMM=float(sh[-1]),
                    gap_dlam=float((lh[-1] - lf[-1]) / (lf[-1] - 1.0)) if case != "U" else float("nan"),
                    gap_sigma=(float(jnp.max(jnp.abs(sh - sf)) / jnp.abs(sf[0])) if case == "U"
                               else float((sh[-1] - sf[-1]) / sf[-1]) if case == "F" else float("nan")),
                ))
    write_csv(rows, RESULTS / "maes_sweep.csv")

    t = jnp.arange(1, n_steps + 1) * ds
    def series(case, key):
        return {f"{m} {v}": r[key] for (m, c, v), r in out.items() if c == case}
    plotting.panels([
        dict(t=t, series=series("U", "sigma_driven"), title="Case U", ylabel="sigma_yy (MPa)"),
        dict(t=t, series=series("S", "lam_driven"), title="Case S", ylabel="lam_y (-)", legend=False),
        dict(t=t, series=series("F", "sigma_driven"), title="Case F", ylabel="sigma_yy (MPa)", legend=False),
        dict(t=t, series=series("F", "lam_driven"), title="Case F", ylabel="lam_y (-)", legend=False),
    ], RESULTS / "maes_fig2.png")

    for r in rows:
        g = r["gap_sigma"] if r["case"] != "S" else r["gap_dlam"]
        extra = f"  dlam gap {100 * r['gap_dlam']:+6.2f}%" if r["case"] == "F" else ""
        print(f"  {r['model']} {r['case']} {r['variant']:9s}  gap {100 * g:+6.2f}%{extra}")


# ---------- parameter sweep, single compressible matrix ----------

SWEEP_BASE = dict(C10=0.305, K=6.1, k_plus=0.1, g=1.2)
DIAGNOSTIC = {"U": "sigma", "S": "lam", "F": "sigma"}


def matrix_specs(p):
    """Single matrix, G = diag(g^-1/2, g, g^-1/2)"""
    return [Spec(*neo_hookean(p["C10"], p["K"]), G_matrix(p["g"], p["g"] ** -0.5), 1.0, p["k_plus"])]


def reference_bc(case, specs, ds, variant, stretch=1.2, increase=0.2):
    """U: stretch; S/F: (1 + increase) x the variant's own homeostatic stress/force."""
    if case == "U":
        return solver.BC("U", stretch)
    build, module = VARIANTS[variant]
    mix = build(specs, ds, None)
    F, _, _, _ = solver.solve_F(mix, solver.BC("U", 1.0), module.sigma_solver, jnp.ones(1))
    sigma, _ = module.sigma_solver(mix, F)
    s = float(sigma[DRIVEN, DRIVEN])
    P = float(jnp.linalg.det(F) * s / F[DRIVEN, DRIVEN] * 2500.0)
    return solver.BC(case, (s if case == "S" else P) * (1 + increase))


def sweep(total_days=400.0):
    grid = []
    for case in ("U", "S", "F"):
        grid += [(dict(SWEEP_BASE, g=g), 10.0, case) for g in (1.1, 1.2, 1.3)]
        grid += [(dict(SWEEP_BASE, k_plus=k), 10.0, case) for k in (0.05, 0.2)]
        grid += [(dict(SWEEP_BASE), ds, case) for ds in (5.0, 20.0)]

    variants = {k: VARIANTS[k] for k in ("FCMM", "HCMM dep")}
    rows, keep = [], {}
    for p, ds, case in grid:
        n_steps = int(round(total_days / ds))
        specs = matrix_specs(p)
        out = {}
        for var in variants:
            bc = reference_bc(case, specs, ds, var)
            out[var] = run_variants(specs, bc, ds, n_steps, variants={var: variants[var]})[var]
        f, h = out["FCMM"], out["HCMM dep"]
        lam_key = "lam_free" if case == "U" else "lam_driven"
        sf, sh, lf, lh = f["sigma_driven"], h["sigma_driven"], f[lam_key], h[lam_key]
        row = dict(case=case, g=p["g"], k_plus=p["k_plus"], ds=ds, n_steps=n_steps,
                   diagnostic=DIAGNOSTIC[case],
                   sigma_FCMM=float(sf[-1]), sigma_HCMM=float(sh[-1]),
                   lam_FCMM=float(lf[-1]), lam_HCMM=float(lh[-1]),
                   gap_sigma_final=float(jnp.abs(sf[-1] - sh[-1]) / jnp.abs(sf[-1])),
                   gap_lam_final=float(jnp.abs(lf[-1] - lh[-1]) / jnp.abs(lf[-1])))
        rows.append(row)
        gap = row["gap_sigma_final"] if row["diagnostic"] == "sigma" else row["gap_lam_final"]
        print(f"  {case} g={p['g']:.2f} k+={p['k_plus']:.2f} ds={ds:4.1f}  gap[{row['diagnostic']}] "
              f"{100 * gap:5.2f}%", flush=True)
        if (p["g"], p["k_plus"], ds) == (SWEEP_BASE["g"], SWEEP_BASE["k_plus"], 10.0):
            keep[case] = (out, ds, lam_key)

    write_csv(rows, RESULTS / "sweep.csv")
    specs = []
    for case, (out, ds, lam_key) in sorted(keep.items()):
        t = jnp.arange(1, len(out["FCMM"]["sigma_driven"]) + 1) * ds
        specs.append(dict(t=t, series={k: v["sigma_driven"] for k, v in out.items()},
                          title=f"Case {case}", ylabel="sigma_yy (MPa)"))
        specs.append(dict(t=t, series={k: v[lam_key] for k, v in out.items()},
                          title=f"Case {case}", ylabel=lam_key))
    plotting.panels(specs, RESULTS / "trajectories.png")
    plotting.bars({c: [(f"g={r['g']:.1f}\nk={r['k_plus']:.2f}\nds={r['ds']:.0f}",
                        100 * (r["gap_sigma_final"] if r["diagnostic"] == "sigma" else r["gap_lam_final"]))
                       for r in rows if r["case"] == c] for c in ("U", "S", "F")},
                  RESULTS / "gaps.png", ylabel="FCMM-HCMM gap [%]", title="Parameter sweep")


# ---------- single model/case comparison ----------

def compare(model="E", case="S", ds=10.0, n_steps=100):
    out = run_variants(setups.maes_specs(model, case), setups.bc_for(model, case), ds, n_steps,
                       MODELS[model]["ag"])
    for var, r in out.items():
        print(f"  {var:9s}  sigma_yy {float(r['sigma_driven'][-1]):+.5f}  lam_y {float(r['lam_driven'][-1]):.5f}"
              f"  lam_x {float(r['lam_free'][-1]):.5f}")
    t = jnp.arange(1, n_steps + 1) * ds
    plotting.panels([
        dict(t=t, series={k: v["sigma_driven"] for k, v in out.items()}, title=f"{model} {case}", ylabel="sigma_yy (MPa)"),
        dict(t=t, series={k: v["lam_driven"] for k, v in out.items()}, title=f"{model} {case}", ylabel="lam_y (-)"),
        dict(t=t, series={k: v["lam_free"] for k, v in out.items()}, title=f"{model} {case}", ylabel="lam_x (-)"),
    ], RESULTS / f"compare_{model}{case}.png", ncols=3)


# ---------- code verification, constant F, k = 0 ----------

VERIFY_T, VERIFY_DAYS = setups.T_DAYS, 1000.0
VERIFY_FIBER = dict(k1=0.578, k2=1.23, M=(0.0, 1.0, 0.0))


def exact_cmm(mat, F, G, t):
    """sigma(t) = rho/J [e^(-t/T) sigma(F G) + (1 - e^(-t/T)) sigma(G)],  rho = 1   (Eq. 4, 5, 8)"""
    w = jnp.exp(-t / VERIFY_T)[:, None, None]
    return (w * mat.sigma(F @ G) + (1 - w) * mat.sigma(G)) / jnp.linalg.det(F)


def constant_F_run(spec, F, ds, variant):
    """Stress history under prescribed F; FCMM at t = k ds, HCMM at t = (k-1) ds."""
    build, module = VARIANTS[variant]
    mix = build([spec], ds, None)
    out = []
    for _ in range(int(VERIFY_DAYS / ds)):
        s, aux = module.sigma_solver(mix, F)
        mix = module.commit(mix, F, aux)
        out.append(s)
    k = jnp.arange(len(out)) + (1 if variant == "FCMM" else 0)
    return jnp.stack(out), k * ds


def rel_err(sig, ref):
    """max_t |sigma_yy - ref_yy| / |ref_yy(0+) - ref_yy(inf)|"""
    scale = jnp.abs(ref[0, DRIVEN, DRIVEN] - ref[-1, DRIVEN, DRIVEN])
    scale = jnp.where(scale < 1e-12, jnp.abs(ref[0, DRIVEN, DRIVEN]), scale)
    return float(jnp.max(jnp.abs(sig[:, DRIVEN, DRIVEN] - ref[:, DRIVEN, DRIVEN])) / scale)


def verify_spec(kind, g):
    if kind == "matrix":
        return Spec(*neo_hookean(0.305, 6.10), G_matrix(g, g ** -0.5), 1.0)
    return Spec(*fung(VERIFY_FIBER["k1"], VERIFY_FIBER["k2"], VERIFY_FIBER["M"]),
                G_fiber(g, VERIFY_FIBER["M"]), 1.0)


def verify():
    print("homeostasis at F = I, 1000 days: max |sigma(t) - sigma(0)|  (FCMM / HCMM dep)")
    for kind, g in (("matrix", 1.2), ("fiber", 1.1)):
        spec = verify_spec(kind, g)
        d = [float(jnp.max(jnp.abs(s - s[0])))
             for s, _ in (constant_F_run(spec, jnp.eye(3), 10.0, v) for v in ("FCMM", "HCMM dep"))]
        print(f"  {kind:6s} g={g}: {d[0]:.2e} / {d[1]:.2e}")

    print("\nconstant F after jump, error vs exact CMM (relative to relaxation amplitude)")
    for kind, g, lam in (("matrix", 1.0, 1.2), ("matrix", 1.2, 1.2), ("fiber", 1.1, 1.2),
                         ("matrix", 1.2, 1.001), ("fiber", 1.1, 1.001)):
        spec = verify_spec(kind, g)
        F = jnp.diag(jnp.array([1.0, lam, 1.0 / lam]))
        cells = []
        for ds in (10.0, 5.0, 2.5):
            e = [rel_err(s, exact_cmm(spec.mat_f, F, spec.G, t))
                 for s, t in (constant_F_run(spec, F, ds, v) for v in ("FCMM", "HCMM dep"))]
            cells.append(f"ds={ds:4.1f} F {100 * e[0]:5.2f}% H {100 * e[1]:5.2f}%")
        print(f"  {kind:6s} g={g} lam={lam}:  " + "  ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["maes", "sweep", "compare", "verify"])
    ap.add_argument("args", nargs="*", help="compare: MODEL CASE")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.study == "compare":
        compare(*a.args)
    else:
        globals()[a.study]()
