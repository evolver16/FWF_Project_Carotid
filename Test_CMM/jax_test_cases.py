"""FCMM vs HCMM on the same single-element problem.

Both models expose the same entry point, so jaxFEM_solver drives both without
knowing which is which:

    sigma_tot, aux = model.sigma_solver(state, F)
    state          = model.commit(state, F, aux)

Constituents are built here from one shared spec so the two models get exactly
the same W^j, G^j, T^j and gains.

sigma_f^elas(0) = 0 here, and that is the paper's intended mode, not an
accident: a neo-Hookean's isochoric stress is deviatoric, so tr(sigma(G)) = 0
for an isochoric G. Eqs. 11/13/24 then omit the denominator and the gain
carries units of MPa^-1 (Table 2, footnote 2). Both models must use the SAME
omitted-denominator branch for the comparison to mean anything -- see
FCMM.sigma_f_rel and HCMM._sigma_f_rel.

A single fiber along the driven axis would leave sigma_22 identically zero, so
the equilibrium residual on the free axis would be satisfied by any stretch and
its Jacobian singular; the matrix supplies that stiffness.

The one asymmetry is burn-in. FCMM integrates over a cohort history that starts
empty, so its first step returns only ~ds/T of the homeostatic stress and it
needs ~7T at F=I with the gains off to fill (see CMM_test_cases.burn_in). HCMM
carries no cohort history -- its state is just (rho, F_r) -- so it is
equilibrated by construction.

Naming: CMM is the family, FCMM the full model, HCMM the homogenized one.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

import FCMM
import HCMM
import jaxFEM_solver as fem
from CMM_test_cases import burn_in
from CMM_plotter import plot_model_comparison

DRIVEN_AXIS = fem.DRIVEN_AXIS

# Maes & Famaey Table 2, models D/E: compressible matrix plus one fiber family.
# phi_0 is set equal to rho_0 so that FCMM's sigma_f(0) = (rho_0/phi_0) tr(sigma(G))
# matches HCMM's tr(sigma(G)).
MATRIX = dict(C10=0.305, K=6.1, T=101.0, k_plus=0.1, k_minus=0.0,
              rho_0=1.0, g=1.2)
FIBER = dict(k1=0.0289, k2=1.23, T=101.0, k_plus=0.1, k_minus=0.0,
             rho_0=0.2, g=1.1, M=(0.0, 1.0, 0.0))


def G_matrix(g):
    """G^elas = diag(g^-1/2, g, g^-1/2) with g on the driven axis, det G = 1.

    Isochoric, as in FCMM.prestress_stress_snapshot and the paper's sec. 2.5.
    """
    d = [1.0 / jnp.sqrt(g)] * 3
    d[DRIVEN_AXIS] = g
    return jnp.diag(jnp.array(d))


def G_fiber(g, M):
    """G^coll = g M(x)M + g^-1/2 (I - M(x)M)                          (Eq. 32)"""
    M = jnp.asarray(M) / jnp.linalg.norm(jnp.asarray(M))
    P = jnp.outer(M, M)
    return g * P + (1.0 / jnp.sqrt(g)) * (jnp.eye(3) - P)


# ==========================================
# Matched builders
# ==========================================


def n_max_for(ds, tol=1e-3):
    """FCMM's cohort buffer is a rolling window, sized by decay rather than by
    run length, so runs of any length reuse the same window."""
    return FCMM.window_for(MATRIX["T"], ds, tol) + 2


WITH_FIBER = True


def _specs(with_fiber=None):
    """(HCMM material, FCMM material, G, params) per constituent, shared by
    both models so they see identical W^j, G^j, T^j and gains."""
    specs = [
        (HCMM.NeoHookean(MATRIX["C10"], MATRIX["K"]),
         FCMM.NeoHookean(MATRIX["C10"], MATRIX["K"]),
         G_matrix(MATRIX["g"]), MATRIX),
    ]
    if WITH_FIBER if with_fiber is None else with_fiber:
        specs.append(
            (HCMM.Fung(FIBER["k1"], FIBER["k2"], jnp.asarray(FIBER["M"])),
             FCMM.Fung(FIBER["k1"], FIBER["k2"], jnp.asarray(FIBER["M"])),
             G_fiber(FIBER["g"], FIBER["M"]), FIBER))
    return specs


def build_FCMM(ds, n_steps=None, burn=True):
    n_max = n_max_for(ds)
    cs = []
    for _, mat_f, G, p in _specs():
        par = FCMM.params(material=mat_f, T=p["T"], G=G, k_minus=p["k_minus"],
                          k_plus=p["k_plus"], phi_0=p["rho_0"], grows=True)
        sigma_f_0 = mat_f.sigma_f(mat_f.sigma(G))   # per unit mass, as HCMM
        cs.append(FCMM.constituent.allocate(par, p["rho_0"], sigma_f_0, n_max, ds))
    mix = FCMM.mixture.allocate(cs, jnp.eye(3), ds, n_max)
    return burn_in(mix) if burn else mix


def build_HCMM(ds, n_steps=None):
    cs = [HCMM.constituent(mat_h, T=p["T"], rho_0=p["rho_0"],
                           k_sigma_plus=p["k_plus"], k_sigma_minus=p["k_minus"],
                           G=G, phi=p["rho_0"])
          for mat_h, _, G, p in _specs()]
    return HCMM.mixture(cs, ds=ds)


MODELS = {"FCMM": (build_FCMM, FCMM), "HCMM": (build_HCMM, HCMM)}


# ==========================================
# Comparison driver
# ==========================================


def compare(bc, n_steps=40, ds=10.0, models=None):
    """Run each model on the same BC through the same solver.

    Returns {label: jaxFEM_solver.run output}.
    """
    models = MODELS if models is None else models
    return {label: fem.run(build(ds, n_steps), module, bc, n_steps=n_steps)
            for label, (build, module) in models.items()}


def time_axis(n_steps, ds):
    """Step k is committed at time k*ds."""
    return jnp.arange(1, n_steps + 1) * ds


def setpoints():
    """sigma_f^j(0) per constituent -- nonzero keeps Eq. 24 out of its
    degenerate branch."""
    return [(("matrix", "fiber")[i], float(m.sigma_f(m.sigma(G))))
            for i, (m, _, G, _) in enumerate(_specs())]


def summarize(results):
    for label, o in results.items():
        print(f"  {label:5s}  sigma_yy {float(o['sigma_driven'][0]):+.6f}"
              f" -> {float(o['sigma_driven'][-1]):+.6f}"
              f"   lam_free {float(o['x'][0, 0]):.6f} -> {float(o['x'][-1, 0]):.6f}")


def plot_comparison(results, n_steps, ds, bc, save_path=None):
    """Driven stress and free lateral stretch, both models on each panel."""
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
    for name, sf0 in setpoints():
        print(f"  sigma_f_0 ({name:6s}) = {sf0:+.6e}")
    results = compare(bc, n_steps=N_STEPS, ds=DS)
    summarize(results)
    plot_comparison(results, N_STEPS, DS, bc, save_path="fcmm_vs_hcmm.png")
