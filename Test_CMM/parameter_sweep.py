"""Does HCMM track FCMM across parameters, or only at one lucky point?

Sweeps the deposition stretch g, the mass-production gain k_sigma_+, the load
case (U/S/F) and the step size ds, running both models through the same
jaxFEM_solver, and reports the relative gap in the driven stress and the free
stretch. Everything lands in results/.

Both models are built from one parameter dict, so they see identical W^j, G^j,
T^j and gains. For cases S and F the target is taken from HCMM's equilibrated
reference state and then applied verbatim to BOTH models, so the boundary
condition is numerically the same for each -- otherwise each would be chasing
its own slightly different target and the comparison would be circular.

sigma_f^elas(0) = 0 throughout: G is isochoric and a neo-Hookean's isochoric
stress is deviatoric, so Eqs. 11/13/24 run with the denominator omitted and the
gain in MPa^-1 (Table 2, footnote 2).

Two things the metric has to respect, or the summary flatters itself:

  * Whichever quantity a load case PRESCRIBES is not evidence. Case S fixes
    sigma_driven, so both models hit it to solver tolerance and the stress gap
    is identically zero; only the free stretch is diagnostic there. Case U
    fixes lam_driven, so the stress is the diagnostic. Each row records which.
  * Runs at different ds must cover the same PHYSICAL time, so
    n_steps = total_days/ds rather than a fixed step count. A fixed count
    silently compares 200 days against 800 days and says nothing about
    discretization (the paper's convergence study, sec. 2.6 / Fig. 3).
"""

import csv
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import FCMM
import HCMM
import jaxFEM_solver as fem
from CMM_test_cases import burn_in
from CMM_plotter import plot_model_comparison

RESULTS = pathlib.Path(__file__).parent / "results"
DRIVEN = fem.DRIVEN_AXIS

BASE = dict(C10=0.305, K=6.1, T=101.0, k_plus=0.1, k_minus=0.0, rho_0=1.0,
            g=1.2)


def G_matrix(g):
    """G^elas = diag(g^-1/2, g, g^-1/2), g on the driven axis, det G = 1."""
    d = [1.0 / jnp.sqrt(g)] * 3
    d[DRIVEN] = g
    return jnp.diag(jnp.array(d))


# ==========================================
# Matched builders
# ==========================================


def build_FCMM(p, ds, burn=True):
    n_max = FCMM.window_for(p["T"], ds) + 2
    mat = FCMM.NeoHookean(p["C10"], p["K"])
    G = G_matrix(p["g"])
    par = FCMM.params(material=mat, T=p["T"], G=G, k_minus=p["k_minus"],
                      k_plus=p["k_plus"], phi_0=p["rho_0"], grows=True)
    c = FCMM.constituent.allocate(par, p["rho_0"],
                                  mat.sigma_f(mat.sigma(G)), n_max, ds)
    mix = FCMM.mixture.allocate([c], jnp.eye(3), ds, n_max)
    return burn_in(mix) if burn else mix


def build_HCMM(p, ds):
    mat = HCMM.NeoHookean(p["C10"], p["K"])
    c = HCMM.constituent(mat, T=p["T"], rho_0=p["rho_0"],
                         k_sigma_plus=p["k_plus"], k_sigma_minus=p["k_minus"],
                         G=G_matrix(p["g"]), phi=p["rho_0"])
    return HCMM.mixture([c], ds=ds)


def reference_state(p, ds, which="HCMM", A0=2500.0):
    """Equilibrated state at lam_driven = 1: returns (sigma_yy, force_yy)."""
    mix = build_HCMM(p, ds) if which == "HCMM" else build_FCMM(p, ds)
    module = HCMM if which == "HCMM" else FCMM
    F, _, _, _ = fem.solve_F(mix, fem.BC("U", 1.0), module.sigma_solver,
                             jnp.ones(1))
    sigma, _ = module.sigma_solver(mix, F)
    J = jnp.linalg.det(F)
    P = J * sigma[DRIVEN, DRIVEN] / F[DRIVEN, DRIVEN]
    return float(sigma[DRIVEN, DRIVEN]), float(P * A0)


def make_bc(case, p, ds, which, stretch=1.2, increase=0.2):
    """Cases S and F need a load level, and there are two defensible choices.

    which="own"    -> 1+increase times THAT model's own homeostatic reference,
                      so both see the same RELATIVE overload.
    which="HCMM"   -> the same absolute target for both.

    They are not equivalent: FCMM's homeostatic stress sits a few percent below
    HCMM's, because its freshly deposited cohort is still at F_e = G while the
    older ones are at F G, and that cohort carries roughly 5% of the trapezoid
    weight. A shared absolute target therefore loads FCMM harder in relative
    terms and inflates the case S/F gap with an offset that has nothing to do
    with the growth and remodeling response.
    """
    if case == "U":
        return fem.BC("U", stretch)
    sigma_ref, force_ref = reference_state(p, ds, which)
    if case == "S":
        return fem.BC("S", sigma_ref * (1.0 + increase))
    return fem.BC("F", force_ref * (1.0 + increase))


# ==========================================
# One comparison
# ==========================================


#: Which gap is meaningful per load case; the other quantity is prescribed by
#: the boundary condition and matches by construction.
DIAGNOSTIC = {"U": "sigma", "S": "lam", "F": "sigma"}


def run_pair(p, ds, case, total_days=400.0, stretch=1.2, target_mode="own"):
    n_steps = int(round(total_days / ds))
    out, bcs = {}, {}
    for label, build, module in (("FCMM", build_FCMM, FCMM),
                                 ("HCMM", build_HCMM, HCMM)):
        which = label if target_mode == "own" else "HCMM"
        bcs[label] = make_bc(case, p, ds, which, stretch=stretch)
        out[label] = fem.run(build(p, ds), module, bcs[label], n_steps=n_steps)
    bc = bcs["HCMM"]

    sf = out["FCMM"]["sigma_driven"]
    sh = out["HCMM"]["sigma_driven"]
    lf = out["FCMM"]["x"][:, 0]
    lh = out["HCMM"]["x"][:, 0]
    scale = float(jnp.max(jnp.abs(sf)))

    row = dict(
        case=case, g=p["g"], k_plus=p["k_plus"], T=p["T"], ds=ds,
        n_steps=n_steps, total_days=total_days,
        diagnostic=DIAGNOSTIC[case], target_mode=target_mode,
        sigma_FCMM=float(sf[-1]), sigma_HCMM=float(sh[-1]),
        lam_FCMM=float(lf[-1]), lam_HCMM=float(lh[-1]),
        gap_sigma_final=float(jnp.abs(sf[-1] - sh[-1]) / jnp.abs(sf[-1])),
        gap_sigma_max=float(jnp.max(jnp.abs(sf - sh)) / scale),
        gap_lam_final=float(jnp.abs(lf[-1] - lh[-1]) / jnp.abs(lf[-1])),
    )
    return row, out, bc


# ==========================================
# Sweep
# ==========================================


def sweep(total_days=400.0):
    rows, keep = [], {}

    grid = []
    for case in ("U", "S", "F"):
        for g in (1.1, 1.2, 1.3):
            grid.append((dict(BASE, g=g), 10.0, case))
        for k_plus in (0.05, 0.2):
            grid.append((dict(BASE, k_plus=k_plus), 10.0, case))
        for ds in (5.0, 20.0):
            grid.append((dict(BASE), ds, case))

    for p, ds, case in grid:
        row, out, bc = run_pair(p, ds, case, total_days=total_days)
        rows.append(row)
        d = (row["gap_sigma_final"] if row["diagnostic"] == "sigma"
             else row["gap_lam_final"])
        print(f"  {case}  g={p['g']:.2f} k+={p['k_plus']:.2f} ds={ds:5.1f} n={row['n_steps']:3d}"
              f"   sigma {row['sigma_FCMM']:+.5f}/{row['sigma_HCMM']:+.5f}"
              f"   lam {row['lam_FCMM']:.4f}/{row['lam_HCMM']:.4f}"
              f"   gap[{row['diagnostic']}] {d*100:6.2f}%", flush=True)
        if (p["g"], p["k_plus"], ds) == (BASE["g"], BASE["k_plus"], 10.0):
            keep[case] = (out, bc, ds)

    return rows, keep


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_trajectories(keep, path):
    """One column per load case: driven stress and free stretch, both models."""
    fig, axes = plt.subplots(2, len(keep), figsize=(6 * len(keep), 8),
                             squeeze=False)
    for col, (case, (out, bc, ds)) in enumerate(sorted(keep.items())):
        t = jnp.arange(1, out["FCMM"]["x"].shape[0] + 1) * ds
        plot_model_comparison(
            t, {k: v["sigma_driven"] for k, v in out.items()},
            "sigma_yy (MPa)", f"Case {case}: driven stress", ax=axes[0][col])
        plot_model_comparison(
            t, {k: v["x"][:, 0] for k, v in out.items()},
            "lam_free (-)", f"Case {case}: free lateral stretch",
            ax=axes[1][col])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gaps(rows, path):
    """Final-stress gap for every sweep point, grouped by load case."""
    fig, ax = plt.subplots(figsize=(11, 5))
    cases = sorted({r["case"] for r in rows})
    colors = {"U": "tab:blue", "S": "tab:orange", "F": "tab:green"}
    x = 0
    ticks, labels = [], []
    for case in cases:
        for r in [r for r in rows if r["case"] == case]:
            gap = (r["gap_sigma_final"] if r["diagnostic"] == "sigma"
                   else r["gap_lam_final"])
            ax.bar(x, gap * 100, color=colors[case])
            ticks.append(x)
            labels.append(f"g={r['g']:.1f}\nk={r['k_plus']:.2f}\nds={r['ds']:.0f}")
            x += 1
        x += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("relative FCMM-HCMM gap in the free quantity  [%]")
    ax.set_title("HCMM vs FCMM across the sweep -- stress for U/F, stretch "
                 "for S (blue U, orange S, green F)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    TOTAL_DAYS = 400.0
    print(f"sweep: {TOTAL_DAYS:.0f} days per run, n_steps = total_days/ds")
    rows, keep = sweep(total_days=TOTAL_DAYS)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(rows, RESULTS / "sweep.csv")
    plot_trajectories(keep, RESULTS / "trajectories.png")
    plot_gaps(rows, RESULTS / "gaps.png")

    diag = sorted((r["gap_sigma_final"] if r["diagnostic"] == "sigma"
                   else r["gap_lam_final"]) for r in rows)
    print()
    print(f"  runs                      : {len(rows)}")
    print(f"  diagnostic gap median/max : {diag[len(diag)//2]*100:.2f}% / {diag[-1]*100:.2f}%")
    print(f"  written to {RESULTS}")
