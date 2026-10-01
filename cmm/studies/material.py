"""Material point (homogeneous F): HCMM vs FCMM vs the exact solution, Maes & Famaey (2023) Fig. 2, parameter sweep,
and the readiness of the models for the FE code (batching, objectivity, tangent, gradients)."""

import time
from dataclasses import replace

import jax
import jax.numpy as jnp

import materials
import plotting
import setups
from setups import MODELS, TARGETS, VARIANTS, Spec, G_fiber, G_matrix
from studies.common import Log, outputs, write_csv
from verification import material_point

DRIVEN = material_point.DRIVEN_AXIS


def run_variants(specs, bc, ds, n_steps, ag=None, variants=VARIANTS, x0=None):
    """{variant: material_point.run output} for the same specs and BC."""
    return {var: material_point.run(build(specs, ds, ag), module, bc, n_steps=n_steps,
                                    x0=jnp.ones(bc.n_unknowns) if x0 is None else x0)
            for var, (build, module) in variants.items()}


# ---------- Maes & Famaey (2023) Table 1/2, Fig. 2 ----------

def maes_benchmark(ds=10.0, n_steps=100):
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
    csv_path, fig_path = outputs("material/maes_benchmark", "maes.csv", "fig2.png")
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
    return [Spec(materials.NeoHookean(p["C10"], p["K"]), G_matrix(p["g"], p["g"] ** -0.5), 1.0, p["k_plus"])]


def reference_bc(case, specs, ds, variant, stretch=1.2, increase=0.2):
    """U: stretch; S/F: (1 + increase) x the variant's own homeostatic stress/force."""
    if case == "U":
        return material_point.BC("U", stretch)
    build, module = VARIANTS[variant]
    mix = build(specs, ds, None)
    F, _, _, _ = material_point.solve_F(mix, material_point.BC("U", 1.0), module.sigma_solver, jnp.ones(1))
    sigma, _ = module.sigma_solver(mix, F)
    s = float(sigma[DRIVEN, DRIVEN])
    P = float(jnp.linalg.det(F) * s / F[DRIVEN, DRIVEN] * 2500.0)
    return material_point.BC(case, (s if case == "S" else P) * (1 + increase))


def parameter_sweep(total_days=400.0):
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

    csv_path, traj_path, gaps_path = outputs("material/parameter_sweep", "sweep.csv", "trajectories.png", "gaps.png")
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

def case_comparison(model="E", case="S", ds=10.0, n_steps=100):
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
    ], outputs("material/case_comparison", f"{model}{case}.png")[0], ncols=3)


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
        return Spec(materials.NeoHookean(0.305, 6.10), G_matrix(g, g ** -0.5), 1.0)
    return Spec(materials.Fung(VERIFY_FIBER["k1"], VERIFY_FIBER["k2"], jnp.asarray(VERIFY_FIBER["M"])),
                G_fiber(g, VERIFY_FIBER["M"]), 1.0)


def exact_solution():
    log = Log()
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
            e = [rel_err(s, exact_cmm(spec.mat, F, spec.G, t))
                 for s, t in (constant_F_run(spec, F, ds, v) for v in ("FCMM", "HCMM dep"))]
            cells.append(f"ds={ds:4.1f} F {100 * e[0]:5.2f}% H {100 * e[1]:5.2f}%")
        log(f"  {kind:6s} g={g} lam={lam}:  " + "  ".join(cells))
    log.save(outputs("material/exact_solution", "exact_solution.txt")[0])


# ---------- readiness for the FE code: batching, general F, objectivity, tangent, gradients ----------

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
    return [replace(sp, mat=(materials.Fung(sp.mat.k1, sp.mat.k2, Q @ sp.mat.M) if isinstance(sp.mat, materials.Fung)
                             else sp.mat), G=Q @ sp.G @ Q.T)
            for sp in specs]


def stack(states):
    return jax.tree.map(lambda *x: jnp.stack(x), *states)


def err_rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


def fe_readiness(n_points=64, n_steps=5, ds=10.0):
    log = Log()

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
            return [replace(m, mat=setups.matrix_material(v, None if inc else m.mat.K))] + specs[1:]
        if which == "k1":
            return [m] + [replace(sp, mat=materials.Fung(v, sp.mat.k2, sp.mat.M)) for sp in specs[1:]]
        return [replace(sp, k_plus=v) for sp in specs]

    for name, model, build, mod, _ in cases:
        inc, ag = MODELS[model]["inc"], MODELS[model]["ag"]
        base = setups.maes_specs(model, "S")
        params = {"C10": float(base[0].mat.C10), "k_plus": 0.1}
        if len(base) > 1:
            params["k1"] = float(base[1].mat.k1)
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

    log.save(outputs("material/fe_readiness", "fe_readiness.txt")[0])
