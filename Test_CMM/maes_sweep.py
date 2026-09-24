"""Maes & Famaey (2023) Table 1/2 models under cases U/S/F: FCMM vs HCMM (deposition, initial)."""

import csv
import pathlib
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import FCMM
import HCMM
import jaxFEM_solver as fem

RESULTS = pathlib.Path(__file__).parent / "results"
DRIVEN, FREE = fem.DRIVEN_AXIS, fem.FREE_AXIS
DS, N_STEPS, T_COLL = 10.0, 100, 101.0
ALPHA = jnp.pi / 8
AG = jnp.array([0.0, 0.0, 1.0])
FIBER_DIRS = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
              (float(jnp.sin(ALPHA)), float(jnp.cos(ALPHA)), 0.0),
              (-float(jnp.sin(ALPHA)), float(jnp.cos(ALPHA)), 0.0)]

MATRIX_AC = dict(C10=0.305, K=6.10, rho_0=1.0)
MATRIX_DE = dict(C10=0.0305 / 0.8, K=0.610 / 0.8, rho_0=0.8)
FIBER = dict(k1=0.0289 / 0.05, k2=1.23, rho_0=0.05, g=1.1)

MODELS = {
    "A": dict(matrix=MATRIX_AC, inc=True, matrix_remodels=True, k_matrix=0.0, fibers=False, k_fiber=0.0, ag=None),
    "B": dict(matrix=MATRIX_AC, inc=False, matrix_remodels=True, k_matrix=0.0, fibers=False, k_fiber=0.0, ag=None),
    "C": dict(matrix=MATRIX_AC, inc=False, matrix_remodels=True, k_matrix=0.1, fibers=False, k_fiber=0.0, ag=AG),
    "D": dict(matrix=MATRIX_DE, inc=True, matrix_remodels=False, k_matrix=0.0, fibers=True, k_fiber=0.0, ag=None),
    "E": dict(matrix=MATRIX_DE, inc=False, matrix_remodels=False, k_matrix=0.0, fibers=True, k_fiber=0.1, ag=AG),
}
TARGETS = {"U": 1.5, "S": 0.120, "F": 300.0}
CASES = list(TARGETS)


def bc_for(model, case):
    return fem.BC(case, TARGETS[case], hybrid=MODELS[model]["inc"])


def G_fiber(g, M):
    """G^coll = g M(x)M + g^-1/2 (I - M(x)M)"""
    M = jnp.asarray(M) / jnp.linalg.norm(jnp.asarray(M))
    P = jnp.outer(M, M)
    return g * P + (1.0 / jnp.sqrt(g)) * (jnp.eye(3) - P)


def G_matrix(g_y, g_x):
    """G^elas = diag(1/(g_y g_x), g_y, g_x), det G = 1"""
    return jnp.diag(jnp.array([1.0 / (g_y * g_x), g_y, g_x]))


def specs(model, case):
    """(name, HCMM material, FCMM material, G, rho_0, remodels, k_plus) per constituent."""
    m = MODELS[model]
    mat = m["matrix"]
    out = []
    fib = []
    if m["fibers"]:
        for M in FIBER_DIRS:
            G = jnp.eye(3) if case == "U" else G_fiber(FIBER["g"], M)
            fib.append((f"coll{M}", HCMM.Fung(FIBER["k1"], FIBER["k2"], jnp.asarray(M)),
                        FCMM.Fung(FIBER["k1"], FIBER["k2"], jnp.asarray(M)), G,
                        FIBER["rho_0"], True, m["k_fiber"]))
    if m["inc"]:
        mat_h, mat_f = HCMM.NeoHookeanInc(mat["C10"]), FCMM.NeoHookeanInc(mat["C10"])
    else:
        mat_h, mat_f = HCMM.NeoHookean(mat["C10"], mat["K"]), FCMM.NeoHookean(mat["C10"], mat["K"])
    G_e = jnp.eye(3) if case == "U" else prestress_G(mat_f, mat["rho_0"], fib, m["inc"])
    out.append(("elas", mat_h, mat_f, G_e, mat["rho_0"], m["matrix_remodels"], m["k_matrix"]))
    return out + fib


def prestress_G(mat_f, rho_e, fib, inc, target=0.100):
    """G^elas such that sigma_tot(F=I) has sigma_yy = target, sigma_xx = 0 (sec. 2.5).

    inc: G = diag(g^-1/2, g, g^-1/2), sigma - p I with p = sigma_xx.
    """
    sig_fib = sum((rho * mf.sigma(G) for _, _, mf, G, rho, _, _ in fib), jnp.zeros((3, 3)))

    if inc:
        def r(x):
            s = rho_e * mat_f.sigma(G_matrix(x[0], 1.0 / jnp.sqrt(x[0]))) + sig_fib
            return jnp.array([s[DRIVEN, DRIVEN] - s[FREE, FREE] - target])
        x = jnp.array([1.1])
        for _ in range(50):
            x = x - jnp.linalg.solve(jax.jacfwd(r)(x), r(x))
        return G_matrix(x[0], 1.0 / jnp.sqrt(x[0]))

    def r(x):
        s = rho_e * mat_f.sigma(G_matrix(x[0], x[1])) + sig_fib
        return jnp.array([s[DRIVEN, DRIVEN] - target, s[FREE, FREE]])

    x = jnp.array([1.1, 1.0])
    for _ in range(50):
        x = x - jnp.linalg.solve(jax.jacfwd(r)(x), r(x))
    return G_matrix(x[0], x[1])


def burn_in(mix, n_steps):
    saved = [c.params for c in mix.constituents]
    off = jnp.asarray(0.0)
    mix = mix.replace(constituents=[FCMM.constituent(c.params.with_gains(off, off), c.history)
                                    for c in mix.constituents])
    for _ in range(n_steps):
        _, aux = FCMM.sigma_solver(mix, jnp.eye(3))
        mix = FCMM.commit(mix, jnp.eye(3), aux)
    return mix.replace(constituents=[FCMM.constituent(p, c.history)
                                     for p, c in zip(saved, mix.constituents)])


def build_FCMM(model, case):
    n_max = FCMM.window_for(T_COLL, DS) + 2
    cs = []
    for _, _, mat, G, rho, remodels, k in specs(model, case):
        par = FCMM.params(material=mat, T=T_COLL, G=G, k_minus=0.0, k_plus=k,
                          phi_0=rho, grows=True, remodels=remodels)
        cs.append(FCMM.constituent.allocate(par, rho, mat.sigma_f(mat.sigma(G)), n_max, DS))
    mix = FCMM.mixture.allocate(cs, jnp.eye(3), DS, n_max, ag=MODELS[model]["ag"])
    return burn_in(mix, n_max)


def build_HCMM(model, case, mode):
    sp = specs(model, case)
    rho_tot_0 = sum(s[4] for s in sp)
    cs = [HCMM.constituent(mat, T=T_COLL if remodels else jnp.inf, rho_0=rho,
                           k_sigma_plus=k, k_sigma_minus=0.0, G=G,
                           phi=rho / rho_tot_0, sigma_pre_mode=mode)
          for _, mat, _, G, rho, remodels, k in sp]
    return HCMM.mixture(cs, ds=DS, ag=MODELS[model]["ag"])


VARIANTS = {
    "FCMM": (lambda m, c: build_FCMM(m, c), FCMM),
    "HCMM dep": (lambda m, c: build_HCMM(m, c, "deposition"), HCMM),
    "HCMM init": (lambda m, c: build_HCMM(m, c, "initial"), HCMM),
}


def run_all():
    out = {}
    for model in MODELS:
        for case in CASES:
            bc = bc_for(model, case)
            for var, (build, module) in VARIANTS.items():
                t0 = time.time()
                x0 = jnp.ones(bc.n_unknowns)
                r = fem.run(build(model, case), module, bc, n_steps=N_STEPS, x0=x0)
                out[(model, case, var)] = (r["lam_driven"], r["sigma_driven"])
                print(f"  {model:2s} {case} {var:9s}  lam {float(r['lam_driven'][-1]):.4f}"
                      f"  sigma {float(r['sigma_driven'][-1]):+.4f}  ({time.time()-t0:.0f}s)",
                      flush=True)
    return out


def rows(out):
    res = []
    for model in MODELS:
        for case in CASES:
            lf, sf = out[(model, case, "FCMM")]
            for var in ("HCMM dep", "HCMM init"):
                lh, sh = out[(model, case, var)]
                res.append(dict(
                    model=model, case=case, variant=var,
                    lam_FCMM=float(lf[-1]), lam_HCMM=float(lh[-1]),
                    sigma_FCMM=float(sf[-1]), sigma_HCMM=float(sh[-1]),
                    gap_dlam=float((lh[-1] - lf[-1]) / (lf[-1] - 1.0)) if case != "U" else float("nan"),
                    gap_sigma=(float(jnp.max(jnp.abs(sh - sf)) / jnp.abs(sf[0])) if case == "U"
                               else float((sh[-1] - sf[-1]) / sf[-1]) if case == "F"
                               else float("nan")),
                ))
    return res


def plot(out, path):
    t = jnp.arange(1, N_STEPS + 1) * DS
    colors = dict(zip(MODELS, ["tab:green", "tab:purple", "tab:orange", "tab:red", "tab:blue"]))
    styles = {"FCMM": "-", "HCMM dep": "--", "HCMM init": ":"}
    panels = [("U", 1, "Case U: sigma_yy (MPa)"), ("S", 0, "Case S: lam_y (-)"),
              ("F", 1, "Case F: sigma_yy (MPa)"), ("F", 0, "Case F: lam_y (-)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (case, idx, title) in zip(axes.flat, panels):
        for (model, c, var), v in out.items():
            if c == case:
                ax.plot(t, v[idx], color=colors[model], linestyle=styles[var],
                        label=f"{model} {var}")
        ax.set_title(title)
        ax.set_xlabel("Time (days)")
        ax.grid(True)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = run_all()
    res = rows(out)
    with (RESULTS / "maes_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    plot(out, RESULTS / "maes_fig2.png")
    print()
    for r in res:
        g = r["gap_sigma"] if r["case"] != "S" else r["gap_dlam"]
        extra = f"  dlam gap {100*r['gap_dlam']:+6.2f}%" if r["case"] == "F" else ""
        print(f"  {r['model']:2s} {r['case']} {r['variant']:9s}  gap {100*g:+6.2f}%{extra}")
