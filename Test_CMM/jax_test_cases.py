"""FCMM vs HCMM on the same single-element problem.

Both models expose the same entry point, so jaxFEM_solver drives both without
knowing which is which:

    sigma_tot, aux = model.sigma_solver(state, F)
    state          = model.commit(state, F, aux)

The one asymmetry is burn-in. FCMM integrates over a cohort history that starts
empty, so its first step returns only ~ds/T of the homeostatic stress and it
needs ~7T at F=I with the gains off to fill (see CMM_test_cases.burn_in). HCMM
carries no cohort history -- its state is just (rho, F_r) -- so it is
equilibrated by construction and needs no burn-in.

Naming: CMM is the family, FCMM the full model, HCMM the homogenized one.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

import FCMM
import HCMM
import jaxFEM_solver as fem
from CMM_test_cases import build_elastin_matrix, burn_in
from CMM_plotter import plot_model_comparison


# One elastin matrix, shared by both models (Maes & Famaey Table 2, model B/C).
ELASTIN = dict(C10=0.305, K=6.1, T=101.0, k_plus=0.1, k_minus=0.0,
               rho_0=1.0, g=1.2)


def G_elastin(g):
    """Isochoric deposition stretch with the prestretch on the driven (y) axis,
    matching build_elastin_matrix(isochoric=True)."""
    return jnp.diag(jnp.array([1.0 / jnp.sqrt(g), g, 1.0 / jnp.sqrt(g)]))


# ==========================================
# Matched builders
# ==========================================


def build_FCMM(ds, burn=True):
    p = ELASTIN
    c = build_elastin_matrix(g=p["g"], T=p["T"], k_minus=p["k_minus"],
                             k_plus=p["k_plus"], C10=p["C10"], K=p["K"],
                             rho_0=p["rho_0"], phi_0=1.0, grows=True)
    mix = FCMM.mixture([c], jnp.eye(3), ds=ds)
    if burn:
        burn_in(mix)
    return mix


def build_HCMM(ds):
    p = ELASTIN
    material = HCMM.NeoHookean(C10=p["C10"], K=p["K"])
    c = HCMM.constituent(material, T=p["T"], rho_0=p["rho_0"],
                         k_sigma_plus=p["k_plus"], k_sigma_minus=p["k_minus"],
                         G=G_elastin(p["g"]))
    return HCMM.mixture([c], ds=ds)


MODELS = {"FCMM": (build_FCMM, FCMM), "HCMM": (build_HCMM, HCMM)}


# ==========================================
# Comparison driver
# ==========================================


def compare(bc, n_steps=40, ds=10.0, models=None):
    """Run each model on the same BC through the same solver.

    Returns {label: jaxFEM_solver.run output}.
    """
    models = MODELS if models is None else models
    out = {}
    for label, (build, module) in models.items():
        out[label] = fem.run(build(ds), module, bc, n_steps=n_steps)
    return out


def time_axis(n_steps, ds):
    """Step k is committed at time k*ds; step 0 of the G&R phase is at ds."""
    return jnp.arange(1, n_steps + 1) * ds


def summarize(results):
    for label, o in results.items():
        print(f"  {label:5s}  sigma_yy {float(o['sigma_driven'][0]):+.6f}"
              f" -> {float(o['sigma_driven'][-1]):+.6f}"
              f"   lam_free {float(o['x'][0, 0]):.6f} -> {float(o['x'][-1, 0]):.6f}")


def plot_comparison(results, n_steps, ds, bc, save_path=None):
    """Two panels: driven stress and free lateral stretch, both models on each."""
    t = time_axis(n_steps, ds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_model_comparison(
        t, {k: v["sigma_driven"] for k, v in results.items()},
        "sigma_yy (MPa)", f"Case {bc.case}: driven stress", ax=axes[0])
    plot_model_comparison(
        t, {k: v["x"][:, 0] for k, v in results.items()},
        "lam_free (-)", f"Case {bc.case}: free lateral stretch", ax=axes[1])

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


if __name__ == "__main__":
    N_STEPS, DS = 40, 10.0
    bc = fem.BC("U", 1.2)

    print(f"case {bc.case}, target {bc.target}, {N_STEPS} steps of {DS} days")
    results = compare(bc, n_steps=N_STEPS, ds=DS)
    summarize(results)
    plot_comparison(results, N_STEPS, DS, bc, save_path="fcmm_vs_hcmm.png")
