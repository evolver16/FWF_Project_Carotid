"""Studies: python studies.py maes | sweep | compare [MODEL CASE] | verify | fem | fe | cylinder | artery [MODE] | grad"""

import argparse
import csv
import pathlib
import time
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import plotting
import setups
import solver
from setups import MODELS, TARGETS, VARIANTS, Spec, G_fiber, G_matrix, neo_hookean, fung

RESULTS = pathlib.Path(__file__).parent / "results"
DRIVEN = solver.DRIVEN_AXIS


def outputs(study, *names):
    """results/<study>/<name>; one shared suffix _1, _2, ... if any name already exists."""
    d = RESULTS / study
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        paths = [d / (name if n == 0 else f"{pathlib.Path(name).stem}_{n}{pathlib.Path(name).suffix}")
                 for name in names]
        if not any(p.exists() for p in paths):
            return paths
        n += 1


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
    csv_path, fig_path = outputs("maes", "maes.csv", "fig2.png")
    write_csv(rows, csv_path)

    t = jnp.arange(1, n_steps + 1) * ds
    def series(case, key):
        return {f"{m} {v}": r[key] for (m, c, v), r in out.items() if c == case}
    plotting.panels([
        dict(t=t, series=series("U", "sigma_driven"), title="Case U", ylabel="sigma_yy (MPa)"),
        dict(t=t, series=series("S", "lam_driven"), title="Case S", ylabel="lam_y (-)", legend=False),
        dict(t=t, series=series("F", "sigma_driven"), title="Case F", ylabel="sigma_yy (MPa)", legend=False),
        dict(t=t, series=series("F", "lam_driven"), title="Case F", ylabel="lam_y (-)", legend=False),
    ], fig_path)

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

    csv_path, traj_path, gaps_path = outputs("sweep", "sweep.csv", "trajectories.png", "gaps.png")
    write_csv(rows, csv_path)
    specs = []
    for case, (out, ds, lam_key) in sorted(keep.items()):
        t = jnp.arange(1, len(out["FCMM"]["sigma_driven"]) + 1) * ds
        specs.append(dict(t=t, series={k: v["sigma_driven"] for k, v in out.items()},
                          title=f"Case {case}", ylabel="sigma_yy (MPa)"))
        specs.append(dict(t=t, series={k: v[lam_key] for k, v in out.items()},
                          title=f"Case {case}", ylabel=lam_key))
    plotting.panels(specs, traj_path)
    plotting.bars({c: [(f"g={r['g']:.1f}\nk={r['k_plus']:.2f}\nds={r['ds']:.0f}",
                        100 * (r["gap_sigma_final"] if r["diagnostic"] == "sigma" else r["gap_lam_final"]))
                       for r in rows if r["case"] == c] for c in ("U", "S", "F")},
                  gaps_path, ylabel="FCMM-HCMM gap [%]", title="Parameter sweep")


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
    ], outputs("compare", f"{model}{case}.png")[0], ncols=3)


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
    lines = []

    def log(text):
        print(text)
        lines.append(text)

    log("homeostasis at F = I, 1000 days: max |sigma(t) - sigma(0)|  (FCMM / HCMM dep)")
    for kind, g in (("matrix", 1.2), ("fiber", 1.1)):
        spec = verify_spec(kind, g)
        d = [float(jnp.max(jnp.abs(s - s[0])))
             for s, _ in (constant_F_run(spec, jnp.eye(3), 10.0, v) for v in ("FCMM", "HCMM dep"))]
        log(f"  {kind:6s} g={g}: {d[0]:.2e} / {d[1]:.2e}")

    log("\nconstant F after jump, error vs exact CMM (relative to relaxation amplitude)")
    for kind, g, lam in (("matrix", 1.0, 1.2), ("matrix", 1.2, 1.2), ("fiber", 1.1, 1.2),
                         ("matrix", 1.2, 1.001), ("fiber", 1.1, 1.001)):
        spec = verify_spec(kind, g)
        F = jnp.diag(jnp.array([1.0, lam, 1.0 / lam]))
        cells = []
        for ds in (10.0, 5.0, 2.5):
            e = [rel_err(s, exact_cmm(spec.mat_f, F, spec.G, t))
                 for s, t in (constant_F_run(spec, F, ds, v) for v in ("FCMM", "HCMM dep"))]
            cells.append(f"ds={ds:4.1f} F {100 * e[0]:5.2f}% H {100 * e[1]:5.2f}%")
        log(f"  {kind:6s} g={g} lam={lam}:  " + "  ".join(cells))
    outputs("verify", "verify.txt")[0].write_text("\n".join(lines) + "\n")


# ---------- FEM readiness: batching, general F, objectivity, tangent, gradients ----------

def random_rotation(key):
    """Q = expm(skew(w)) via Rodrigues, det Q = 1"""
    w = jax.random.normal(key, (3,))
    th = jnp.linalg.norm(w)
    K = jnp.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
    return jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * K @ K


def random_F(key, amp=0.1, isochoric=False):
    F = jnp.eye(3) + amp * jax.random.normal(key, (3, 3))
    return F / jnp.linalg.det(F) ** (1 / 3) if isochoric else F


def rotate_specs(specs, Q):
    """Material frame rotated by Q: M -> Q M, G -> Q G Q^T"""
    out = []
    for sp in specs:
        mats = ((sp.mat_f, sp.mat_h) if not isinstance(sp.mat_f, setups.fcmm.Fung)
                else fung(sp.mat_f.k1, sp.mat_f.k2, Q @ sp.mat_f.M))
        out.append(replace(sp, mat_f=mats[0], mat_h=mats[1], G=Q @ sp.G @ Q.T))
    return out


def stack(states):
    return jax.tree.map(lambda *x: jnp.stack(x), *states)


def err_rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


def fem(n_points=64, n_steps=5, ds=10.0):
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    def status(ok):
        return "PASS" if ok else "FAIL"

    key = jax.random.PRNGKey(0)
    cases = [("HCMM", "E", setups.build_hcmm, setups.hcmm, n_points),
             ("HCMM", "A", setups.build_hcmm, setups.hcmm, n_points),
             ("FCMM", "E", setups.build_fcmm, setups.fcmm, 8)]

    log("a/c) batching over points with own fiber orientation, general F: vmap vs loop")
    for name, model, build, mod, n in cases:
        inc, ag = MODELS[model]["inc"], MODELS[model]["ag"]
        keys = jax.random.split(jax.random.fold_in(key, ord(model)), 2 * n)
        Qs = [random_rotation(k) for k in keys[:n]]
        Fs = jnp.stack([random_F(k, 0.08, inc) for k in keys[n:]])
        base = setups.maes_specs(model, "S")
        states = [build(rotate_specs(base, Q), ds, None if ag is None else Q @ ag) for Q in Qs]
        batched = stack(states)
        v_sigma, v_commit = jax.vmap(mod.sigma_solver), jax.vmap(mod.commit)
        e = 0.0
        for _ in range(n_steps):
            sb, aux = v_sigma(batched, Fs)
            batched = v_commit(batched, Fs, aux)
            ref = []
            for i in range(n):
                si, ai = mod.sigma_solver(states[i], Fs[i])
                states[i] = mod.commit(states[i], Fs[i], ai)
                ref.append(si)
            e = max(e, err_rel(sb, jnp.stack(ref)))
        log(f"  {name} {model}  N={n:3d}  max rel diff {e:.1e}  {status(e < 1e-10)}")

    log("\nb) objectivity: sigma(QF) = Q sigma(F) Q^T over steps")
    for name, model, build, mod, _ in cases:
        inc, ag = MODELS[model]["inc"], MODELS[model]["ag"]
        k1, k2 = jax.random.split(jax.random.fold_in(key, 7 + ord(model)))
        F, Q = random_F(k1, 0.08, inc), random_rotation(k2)
        specs = setups.maes_specs(model, "S")
        a, b = build(specs, ds, ag), build(specs, ds, ag)
        e = 0.0
        for _ in range(n_steps):
            sa, xa = mod.sigma_solver(a, F)
            sb, xb = mod.sigma_solver(b, Q @ F)
            a, b = mod.commit(a, F, xa), mod.commit(b, Q @ F, xb)
            e = max(e, err_rel(sb, Q @ sa @ Q.T))
        log(f"  {name} {model}  max rel diff {e:.1e}  {status(e < 1e-10)}")

    log("\nd) tangent dsigma/dF: forward AD vs central FD (after 3 steps)")
    for name, model, build, mod, _ in cases:
        inc, ag = MODELS[model]["inc"], MODELS[model]["ag"]
        F = random_F(jax.random.fold_in(key, 11 + ord(model)), 0.08, inc)
        st = build(setups.maes_specs(model, "S"), ds, ag)
        for _ in range(3):
            _, x = mod.sigma_solver(st, F)
            st = mod.commit(st, F, x)
        sig = lambda Fx: mod.sigma_solver(st, Fx)[0]
        ad = jax.jacfwd(sig)(F)
        h = 1e-6
        fd = jnp.stack([(sig(F + h * E) - sig(F - h * E)) / (2 * h)
                        for E in jnp.eye(9).reshape(9, 3, 3)], axis=-1).reshape(3, 3, 3, 3)
        e = err_rel(ad, fd)
        log(f"  {name} {model}  rel diff {e:.1e}  {status(e < 1e-6)}")

    log("\ne) d sigma_yy(20 steps) / d parameter through the simulation: reverse AD vs FD")
    F = jnp.diag(jnp.array([1.0, 1.1, 1 / 1.1])) + 0.03 * (jnp.eye(3)[:, [1, 2, 0]])
    F = F / jnp.linalg.det(F) ** (1 / 3)

    def simulate(specs, build, mod, ag):
        st = build(specs, ds, ag)
        s = None
        for _ in range(20):
            s, x = mod.sigma_solver(st, F)
            st = mod.commit(st, F, x)
        return s[DRIVEN, DRIVEN]

    def with_param(specs, which, v, inc):
        m = specs[0]
        if which == "C10":
            mats = neo_hookean(v, None if inc else m.mat_f.K)
            return [replace(m, mat_f=mats[0], mat_h=mats[1])] + specs[1:]
        if which == "k1":
            return [m] + [replace(sp, mat_f=f, mat_h=h_) for sp in specs[1:]
                          for f, h_ in [fung(v, sp.mat_f.k2, sp.mat_f.M)]]
        return [replace(sp, k_plus=v) for sp in specs]

    for name, model, build, mod, _ in cases:
        inc, ag = MODELS[model]["inc"], MODELS[model]["ag"]
        base = setups.maes_specs(model, "S")
        params = {"C10": float(base[0].mat_f.C10), "k_plus": 0.1}
        if len(base) > 1:
            params["k1"] = float(base[1].mat_f.k1)
        for which, v0 in params.items():
            f = lambda v: simulate(with_param(base, which, v, inc), build, mod, ag)
            h = 1e-5 * v0
            fd = float((f(v0 + h) - f(v0 - h)) / (2 * h))
            try:
                ad = float(jax.grad(f)(v0))
                ok = abs(ad - fd) <= 1e-5 * abs(fd) + 1e-12
                e = abs(ad - fd) / max(abs(fd), 1e-12)
                log(f"  {name} {model}  {which:6s} AD {ad:+.6e}  FD {fd:+.6e}  rel {e:.1e}  {status(ok)}")
            except Exception as ex:
                log(f"  {name} {model}  {which:6s} not reverse-differentiable ({type(ex).__name__})")

    outputs("fem", "fem_check.txt")[0].write_text("\n".join(lines) + "\n")


# ---------- FE M1: patch test, single hex8 vs single-element solver ----------

def fe():
    import elastic
    import fem
    import mesh as meshlib

    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    log("1) patch test: 2x2x2 distorted box, u = (F0 - I) X on the boundary")
    F0 = jnp.array([[1.08, 0.05, -0.02], [0.03, 0.95, 0.04], [-0.01, 0.02, 1.03]])
    m = meshlib.box((2, 2, 2), (1.0, 1.0, 1.0), perturb=0.2)
    boundary = np.unique(np.concatenate(list(m.nodes.values())))
    fixed = (3 * boundary[:, None] + np.arange(3)).ravel()
    u_exact = ((np.asarray(F0) - np.eye(3)) @ m.X.T).T.ravel()
    materials = {"elastic": (elastic, elastic.NeoHookean(0.305, 6.1)),
                 "HCMM E": (setups.hcmm, setups.build_hcmm(setups.maes_specs("E", "S"), 10.0, MODELS["E"]["ag"]))}
    for name, (model, state) in materials.items():
        sysm = fem.System(m, model)
        states = fem.broadcast_state(state, m.n_elem)
        u, it = sysm.solve(np.zeros(sysm.n_dof), states, fem.BC(fixed, u_exact[fixed]))
        sig = sysm.stress(jnp.asarray(u), states)
        sig0 = model.sigma_solver(state, F0)[0]
        eu = float(np.abs(u - u_exact).max())
        es = float(jnp.abs(sig - sig0).max() / jnp.abs(sig0).max())
        log(f"  {name:8s} Newton it {it}  max|u - u_exact| {eu:.1e}  max rel sigma err {es:.1e}  "
            f"{'PASS' if eu < 1e-10 and es < 1e-10 else 'FAIL'}")

    log("\n2) one hex8 with the Fig. 1 BCs vs solver.py (HCMM dep, 100 steps of 10 days)")
    L = 50.0
    m = meshlib.box((1, 1, 1), (L, L, L))
    top = m.nodes[(DRIVEN, 1)]
    fixed_base = np.concatenate([3 * m.nodes[(0, -1)] + 0, 3 * m.nodes[(0, 1)] + 0,
                                 3 * m.nodes[(DRIVEN, -1)] + DRIVEN, 3 * m.nodes[(2, -1)] + 2])
    for model_name, case in (("B", "U"), ("B", "S"), ("B", "F"), ("E", "U"), ("E", "S"), ("E", "F")):
        specs, ag = setups.maes_specs(model_name, case), MODELS[model_name]["ag"]
        build, module = VARIANTS["HCMM dep"]
        ref = solver.run(build(specs, 10.0, ag), module, setups.bc_for(model_name, case), n_steps=100)
        target = TARGETS[case]
        if case == "U":
            fixed = np.concatenate([fixed_base, 3 * top + DRIVEN])
            vals = np.concatenate([np.zeros(len(fixed_base)), np.full(len(top), (target - 1) * L)])
            bc, faces = fem.BC(fixed, vals), None
        elif case == "S":
            bc, faces = fem.BC(fixed_base, np.zeros(len(fixed_base)), p=-target), m.faces[(DRIVEN, 1)]
        else:
            f = np.zeros(3 * m.n_nodes)
            f[3 * top + DRIVEN] = target / len(top)
            bc, faces = fem.BC(fixed_base, np.zeros(len(fixed_base)), f_dead=f), None
        sysm = fem.System(m, module, pressure_faces=faces)
        states = fem.broadcast_state(build(specs, 10.0, ag), m.n_elem)
        t0 = time.time()
        _, _, hist = sysm.run(states, [bc] * 100,
                              on_step=lambda u, st: (1 + u[3 * top[0] + DRIVEN] / L,
                                                     float(sysm.stress(jnp.asarray(u), st)[0, 0, DRIVEN, DRIVEN])))
        key, q = ("sigma_driven", 1) if case == "U" else ("lam_driven", 0)
        fe_q = np.array([h[q] for h in hist])
        e = float(np.abs(fe_q - np.asarray(ref[key])).max())
        log(f"  {model_name} {case}  {key}(1000 d) FE {fe_q[-1]:.6f}  1-point {float(ref[key][-1]):.6f}  "
            f"max diff {e:.1e}  ({time.time() - t0:.1f}s)  {'PASS' if e < 1e-8 else 'FAIL'}")

    outputs("fe", "fe_m1.txt")[0].write_text("\n".join(lines) + "\n")


# ---------- FE M2: pressurized quarter cylinder, elastic ----------

def cylinder_bc(m, p):
    """Symmetry u_y = 0 at theta = 0, u_x = 0 at theta = pi/2, u_z = 0 at both ends; pressure p inside."""
    import fem
    fixed = np.concatenate([3 * m.nodes[(1, -1)] + 1, 3 * m.nodes[(1, 1)] + 0,
                            3 * m.nodes[(2, -1)] + 2, 3 * m.nodes[(2, 1)] + 2])
    fixed = np.unique(fixed)
    return fem.BC(fixed, np.zeros(len(fixed)), p=p)


def radial_u(m, u, side):
    """mean u . e_r over the inner (-1) or outer (+1) nodes"""
    idx = m.nodes[(0, side)]
    x = m.X[idx]
    e_r = x[:, :2] / np.linalg.norm(x[:, :2], axis=1, keepdims=True)
    return jnp.mean(jnp.sum(jnp.asarray(u).reshape(-1, 3)[idx, :2] * e_r, axis=1))


def cylinder(r_i=5.0, t=1.3, L=0.22, C10=0.305, K=6.1):
    import elastic
    import fem
    import mesh as meshlib

    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    mat = elastic.NeoHookean(C10, K)
    a, b = r_i, r_i + t
    mu, lam = 2 * C10, K - 4 * C10 / 3
    meshes = [(2, 12, 1), (4, 24, 1), (8, 48, 1), (16, 96, 1)]

    log("1) small pressure vs plane-strain Lame solution u_r = C1 r + C2 / r")
    p = 1e-5
    C1 = p * a ** 2 / (2 * (lam + mu) * (b ** 2 - a ** 2))
    C2 = p * a ** 2 * b ** 2 / (2 * mu * (b ** 2 - a ** 2))
    exact = {-1: C1 * a + C2 / a, 1: C1 * b + C2 / b}
    prev = None
    for n in meshes:
        m = meshlib.quarter_cylinder(r_i, t, L, n)
        sysm = fem.System(m, elastic, pressure_faces=m.faces[(0, -1)])
        u, it = sysm.solve(np.zeros(sysm.n_dof), fem.broadcast_state(mat, m.n_elem), cylinder_bc(m, p))
        e = [abs(radial_u(m, u, s) / exact[s] - 1) for s in (-1, 1)]
        rate = "" if prev is None else f"  ratio {prev / e[0]:.1f}"
        prev = e[0]
        log(f"  n={n}  dofs {sysm.n_dof:6d}  rel err u_r(a) {e[0]:.2e}  u_r(b) {e[1]:.2e}{rate}")

    log("\n2) 15 kPa (nonlinear): u_r(a) under refinement")
    p = 0.015
    vals = []
    for n in meshes + [(8, 60, 4)]:
        m = meshlib.quarter_cylinder(r_i, t, L, n)
        sysm = fem.System(m, elastic, pressure_faces=m.faces[(0, -1)])
        t0 = time.time()
        u, it = sysm.solve(np.zeros(sysm.n_dof), fem.broadcast_state(mat, m.n_elem), cylinder_bc(m, p))
        vals.append(radial_u(m, u, -1))
        log(f"  n={n}  elements {m.n_elem:5d}  u_r(a) {vals[-1]:.8f} mm  Newton it {it}  ({time.time() - t0:.1f}s)")
    d = [abs(vals[i] - vals[i + 1]) for i in range(len(meshes) - 1)]
    log("  successive differences " + "  ".join(f"{x:.2e}" for x in d)
        + "  ratios " + "  ".join(f"{d[i] / d[i + 1]:.1f}" for i in range(len(d) - 1)))

    outputs("fe", "fe_m2_cylinder.txt")[0].write_text("\n".join(lines) + "\n")


# ---------- FE M3: arterial G&R, Maes & Famaey (2023) sec. 2.7 / Fig. 4 ----------

def artery_par(**kw):
    """Model E on the artery: elastin (C10, K), fibers (k1, k2, k_plus, g), axial prestretch g_ax, pressures"""
    from setups import FIBER, MATRIX_DE
    par = dict(C10=MATRIX_DE["C10"], K=MATRIX_DE["K"], k1=FIBER["k1"], k2=FIBER["k2"], k_plus=0.1,
               g=FIBER["g"], g_ax=1.2, p_hom=0.010, p_gr=0.015)
    return {k: jnp.asarray(v, dtype=float) for k, v in {**par, **kw}.items()}


@partial(jax.jit, static_argnames="mode")
def artery_states(G_e, Qloc, par, mode, ds=10.0):
    """HCMM per Gauss point: elastin (no turnover) + fibers along e_theta, e_z, +-alpha; growth along e_r."""
    from setups import ALPHA, FIBER, MATRIX_DE
    hcmm = setups.hcmm

    def make(Ge, Q):
        e_r, e_t, e_z = Q[:, 0], Q[:, 1], Q[:, 2]
        dirs = [e_t, e_z, jnp.cos(ALPHA) * e_t + jnp.sin(ALPHA) * e_z, jnp.cos(ALPHA) * e_t - jnp.sin(ALPHA) * e_z]
        cs = [hcmm.constituent(hcmm.NeoHookean(par["C10"], par["K"]), T=jnp.inf, rho_0=MATRIX_DE["rho_0"],
                               k_sigma_plus=0.0, k_sigma_minus=0.0, G=Ge, sigma_pre_mode=mode)]
        cs += [hcmm.constituent(hcmm.Fung(par["k1"], par["k2"], M), T=setups.T_DAYS, rho_0=FIBER["rho_0"],
                                k_sigma_plus=par["k_plus"], k_sigma_minus=0.0, G=G_fiber(par["g"], M),
                                sigma_pre_mode=mode) for M in dirs]
        return hcmm.mixture(cs, ds=ds, ag=e_r)

    n_el = G_e.shape[0]
    st = jax.vmap(jax.checkpoint(make))(G_e.reshape(-1, 3, 3), Qloc.reshape(-1, 3, 3))
    return jax.tree.map(lambda x: x.reshape((n_el, 8) + x.shape[1:]), st)


def artery_mesh(n=(8, 60, 4)):
    """(mesh, System, local bases (e_r, e_theta, e_z) at the Gauss points)"""
    import fem
    import mesh as meshlib
    m = meshlib.quarter_cylinder(n=n)
    sysm = fem.System(m, setups.hcmm, pressure_faces=m.faces[(0, -1)])
    return m, sysm, jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))


def wall_area(m, u):
    """4 x shoelace area of the deformed end face z = 0"""
    q = (jnp.asarray(m.X) + jnp.asarray(u).reshape(-1, 3))[m.faces[(2, -1)]][:, :, :2]
    x, y = q[..., 0], q[..., 1]
    return 2 * jnp.abs(jnp.sum(x * jnp.roll(y, -1, 1) - jnp.roll(x, -1, 1) * y, axis=1)).sum()


def artery_simulate(m, sysm, Qloc, par, mode="deposition", n_steps=100, pre_tol=1e-6, pre_iter=60,
                    tol=1e-10, log=None):
    """Prestress G_elas <- F G_elas at p_hom until u = 0 (Maes UMAT_DEP), G&R at p_gr
    -> (lambda_theta(inner), wall area) per step, reverse-differentiable in par"""
    g_ax = par["g_ax"]
    G_e = Qloc @ jnp.diag(jnp.stack([g_ax ** -0.5, g_ax ** -0.5, g_ax])) @ jnp.swapaxes(Qloc, -1, -2)
    zero = jnp.zeros(sysm.n_dof)
    for it in range(1, pre_iter + 1):
        states = artery_states(G_e, Qloc, par, mode)
        u = sysm.equilibrium(states, cylinder_bc(m, par["p_hom"]), zero, tol)
        du = float(np.abs(sysm.last_u).max())
        if log and (it % 5 == 0 or du < pre_tol):
            log(f"  iteration {it:2d}  max|u| {du:.2e} mm  (Newton {sysm.last_iterations})")
        if du < pre_tol:
            break
        G_e = sysm.gauss_F(u) @ G_e

    a = float(np.linalg.norm(m.X[m.nodes[(0, -1)][0], :2]))
    _, _, hist = sysm.run(states, [cylinder_bc(m, par["p_gr"])] * n_steps, u0=zero, tol=tol,
                          on_step=lambda u, _: (1 + radial_u(m, u, -1) / a, wall_area(m, u)))
    return jnp.stack([h[0] for h in hist]), jnp.stack([h[1] for h in hist])


def artery(mode="deposition", n=(8, 60, 4), n_steps=100):
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    m, sysm, Qloc = artery_mesh(n)
    par = artery_par()
    log(f"mesh {n}: {m.n_elem} hex8, {sysm.n_dof} dofs, HCMM model E, sigma_pre mode '{mode}'")
    log(f"prestress at {float(par['p_hom']) * 1e3:.0f} kPa, then G&R at {float(par['p_gr']) * 1e3:.0f} kPa, "
        f"{n_steps} steps of 10 days   (reference area {float(wall_area(m, np.zeros(sysm.n_dof))):.3f} mm^2)")
    t0 = time.time()
    lam, area = map(np.asarray, artery_simulate(m, sysm, Qloc, par, mode, n_steps, log=log))
    for k in (0, 9, 24, 49, 74, 99):
        if k < n_steps:
            log(f"  day {10 * (k + 1):4d}  lambda_theta(inner) {lam[k]:.4f}  area {area[k]:.3f} mm^2")
    log(f"  done in {time.time() - t0:.1f}s")

    tag = "dep" if mode == "deposition" else "init"
    txt, csv_path, fig = outputs("artery", f"artery_{tag}.txt", f"artery_{tag}.csv", f"artery_{tag}.png")
    write_csv([dict(day=10 * (k + 1), lambda_theta=float(lam[k]), area=float(area[k])) for k in range(n_steps)], csv_path)
    t = np.arange(1, n_steps + 1) * 10.0
    plotting.panels([dict(t=t, series={f"HCMM {tag}": lam}, title="Inner circumferential stretch", ylabel="lambda_theta (-)"),
                     dict(t=t, series={f"HCMM {tag}": area}, title="Cross-sectional area", ylabel="area (mm^2)")], fig)
    txt.write_text("\n".join(lines) + "\n")


# ---------- FE M4: gradients through prestress + G&R ----------

def grad(n_check=(2, 12, 1), steps_check=20, n_full=(8, 60, 4), steps_full=100):
    """Adjoint gradients of lambda_theta(end) vs central differences, cost on the full mesh, identification."""
    import scipy.optimize as so
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    names = ["C10", "K", "k1", "k2", "k_plus", "g", "g_ax", "p_gr"]
    m, sysm, Qloc = artery_mesh(n_check)
    sim = lambda par: artery_simulate(m, sysm, Qloc, par, n_steps=steps_check, pre_tol=0.0, pre_iter=30, tol=1e-12)
    par = artery_par()

    log(f"1) d lambda_theta(day {10 * steps_check}) / d par, mesh {n_check}, 30 prestress iterations: adjoint vs central FD")
    t0 = time.time()
    val, g = jax.value_and_grad(lambda p: sim(p)[0][-1])(par)
    log(f"  lambda_theta {float(val):.8f}   value + gradient in {time.time() - t0:.1f}s")
    for k in names:
        h = 1e-5 * abs(float(par[k]))
        fd = (sim({**par, k: par[k] + h})[0][-1] - sim({**par, k: par[k] - h})[0][-1]) / (2 * h)
        err = abs(float(g[k]) - float(fd)) / max(abs(float(fd)), 1e-12)
        log(f"  {k:7s} adjoint {float(g[k]): .8e}   FD {float(fd): .8e}   rel err {err:.1e}  "
            f"{'PASS' if err < 1e-5 else 'FAIL'}")

    log(f"\n2) identification of (k1, k_plus) from lambda_theta(t), mesh {n_check}, L-BFGS in log space")
    true = artery_par()
    data = sim(true)[0]
    x_ref = {k: true[k] for k in ("k1", "k_plus")}

    def loss(x):
        p = {**true, **{k: x_ref[k] * jnp.exp(x[i]) for i, k in enumerate(x_ref)}}
        return 1e6 * jnp.sum((sim(p)[0] - data) ** 2)

    vg = jax.value_and_grad(loss)
    trace = []

    def fun(x):
        v, dv = vg(jnp.asarray(x))
        trace.append(float(v))
        return float(v), np.asarray(dv)

    x0 = np.log([0.7, 0.5])
    t0 = time.time()
    res = so.minimize(fun, x0, jac=True, method="L-BFGS-B", options=dict(ftol=1e-16, gtol=1e-10, maxiter=40))
    est = {k: float(x_ref[k]) * np.exp(res.x[i]) for i, k in enumerate(x_ref)}
    log(f"  start k1 {float(true['k1']) * 0.7:.4f}  k_plus {float(true['k_plus']) * 0.5:.4f}   loss {trace[0]:.3e}")
    log(f"  found k1 {est['k1']:.6f}  k_plus {est['k_plus']:.6f}   loss {res.fun:.3e}   "
        f"(true {float(true['k1']):.6f}, {float(true['k_plus']):.6f})   {len(trace)} evaluations, {time.time() - t0:.1f}s")

    log(f"\n3) cost on mesh {n_full}, {steps_full} steps")
    m, sysm, Qloc = artery_mesh(n_full)
    run = lambda par: artery_simulate(m, sysm, Qloc, par, n_steps=steps_full)[0][-1]
    t0 = time.time()
    val = run(par)
    t_fwd = time.time() - t0
    t0 = time.time()
    val, g = jax.value_and_grad(run)(par)
    t_grad = time.time() - t0
    log(f"  forward {t_fwd:.1f}s   value + gradient w.r.t. {len(par)} parameters {t_grad:.1f}s   "
        f"(central FD would need {2 * len(par)} forward runs, ~{2 * len(par) * t_fwd:.0f}s)")
    log(f"  lambda_theta(day {10 * steps_full}) {float(val):.6f}")
    for k in par:
        log(f"  d/d{k:7s} {float(g[k]): .6e}   elasticity (p/lam) d lam/dp {float(g[k] * par[k] / val): .4f}")

    outputs("fe", "fe_m4_gradients.txt")[0].write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["maes", "sweep", "compare", "verify", "fem", "fe", "cylinder", "artery", "grad"])
    ap.add_argument("args", nargs="*", help="compare: MODEL CASE")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.study == "compare":
        compare(*a.args)
    elif a.study == "artery":
        artery(*a.args)
    else:
        globals()[a.study]()
