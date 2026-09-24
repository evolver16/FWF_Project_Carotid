"""Deposition stretches, prestress, builders and the Maes & Famaey (2023) Table 1/2 models."""

import pathlib
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(__file__).parent / "compiled"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp

import fcmm
import hcmm
import solver

DRIVEN, FREE = solver.DRIVEN_AXIS, solver.FREE_AXIS
T_DAYS = 101.0


@dataclass
class Spec:
    """One constituent, with matching FCMM and HCMM materials."""
    mat_f: object
    mat_h: object
    G: jnp.ndarray
    rho_0: float
    k_plus: float = 0.0
    remodels: bool = True
    T: float = T_DAYS


def G_matrix(g_y, g_x):
    """G = diag(1/(g_y g_x), g_y, g_x), det G = 1"""
    return jnp.diag(jnp.array([1.0 / (g_y * g_x), g_y, g_x]))


def G_fiber(g, M):
    """G = g M(x)M + g^-1/2 (I - M(x)M)"""
    M = jnp.asarray(M) / jnp.linalg.norm(jnp.asarray(M))
    P = jnp.outer(M, M)
    return g * P + (1.0 / jnp.sqrt(g)) * (jnp.eye(3) - P)


def neo_hookean(C10, K=None):
    """(FCMM, HCMM) matrix material; K=None -> incompressible (Eq. 32)"""
    if K is None:
        return fcmm.NeoHookeanInc(C10), hcmm.NeoHookeanInc(C10)
    return fcmm.NeoHookean(C10, K), hcmm.NeoHookean(C10, K)


def fung(k1, k2, M):
    """(FCMM, HCMM) fiber material"""
    M = jnp.asarray(M)
    return fcmm.Fung(k1, k2, M), hcmm.Fung(k1, k2, M)


def burn_in(mix, n_steps):
    """Fill the FCMM cohort history at F = I with gains off."""
    saved = [c.params for c in mix.constituents]
    off = jnp.asarray(0.0)
    mix = mix.replace(constituents=[fcmm.constituent(c.params.with_gains(off, off), c.history)
                                    for c in mix.constituents])
    for _ in range(n_steps):
        _, aux = fcmm.sigma_solver(mix, jnp.eye(3))
        mix = fcmm.commit(mix, jnp.eye(3), aux)
    return mix.replace(constituents=[fcmm.constituent(p, c.history)
                                     for p, c in zip(saved, mix.constituents)])


def build_fcmm(specs, ds, ag=None):
    n_max = fcmm.window_for(T_DAYS, ds) + 2
    cs = []
    for s in specs:
        par = fcmm.params(material=s.mat_f, T=s.T, G=s.G, k_minus=0.0, k_plus=s.k_plus,
                          phi_0=s.rho_0, grows=True, remodels=s.remodels)
        cs.append(fcmm.constituent.allocate(par, s.rho_0, s.mat_f.sigma_f(s.mat_f.sigma(s.G)),
                                            n_max, ds))
    mix = fcmm.mixture.allocate(cs, jnp.eye(3), ds, n_max, ag=ag)
    return burn_in(mix, n_max)


def build_hcmm(specs, ds, ag=None, mode="deposition"):
    rho_tot_0 = sum(s.rho_0 for s in specs)
    cs = [hcmm.constituent(s.mat_h, T=s.T if s.remodels else jnp.inf, rho_0=s.rho_0,
                           k_sigma_plus=s.k_plus, k_sigma_minus=0.0, G=s.G,
                           phi=s.rho_0 / rho_tot_0, sigma_pre_mode=mode)
          for s in specs]
    return hcmm.mixture(cs, ds=ds, ag=ag)


VARIANTS = {
    "FCMM": (lambda sp, ds, ag: build_fcmm(sp, ds, ag), fcmm),
    "HCMM dep": (lambda sp, ds, ag: build_hcmm(sp, ds, ag, "deposition"), hcmm),
    "HCMM init": (lambda sp, ds, ag: build_hcmm(sp, ds, ag, "initial"), hcmm),
}


# Maes & Famaey (2023) Table 1/2, code axes (Z, Y, X)
ALPHA = jnp.pi / 8
AG_X = jnp.array([0.0, 0.0, 1.0])
FIBER_DIRS = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
              (float(jnp.sin(ALPHA)), float(jnp.cos(ALPHA)), 0.0),
              (-float(jnp.sin(ALPHA)), float(jnp.cos(ALPHA)), 0.0)]
MATRIX_AC = dict(C10=0.305, K=6.10, rho_0=1.0)
MATRIX_DE = dict(C10=0.0305 / 0.8, K=0.610 / 0.8, rho_0=0.8)
FIBER = dict(k1=0.0289 / 0.05, k2=1.23, rho_0=0.05, g=1.1)
MODELS = {
    "A": dict(matrix=MATRIX_AC, inc=True, matrix_remodels=True, k_matrix=0.0, fibers=False, k_fiber=0.0, ag=None),
    "B": dict(matrix=MATRIX_AC, inc=False, matrix_remodels=True, k_matrix=0.0, fibers=False, k_fiber=0.0, ag=None),
    "C": dict(matrix=MATRIX_AC, inc=False, matrix_remodels=True, k_matrix=0.1, fibers=False, k_fiber=0.0, ag=AG_X),
    "D": dict(matrix=MATRIX_DE, inc=True, matrix_remodels=False, k_matrix=0.0, fibers=True, k_fiber=0.0, ag=None),
    "E": dict(matrix=MATRIX_DE, inc=False, matrix_remodels=False, k_matrix=0.0, fibers=True, k_fiber=0.1, ag=AG_X),
}
TARGETS = {"U": 1.5, "S": 0.120, "F": 300.0}


def bc_for(model, case):
    return solver.BC(case, TARGETS[case], hybrid=MODELS[model]["inc"])


def prestress_G(mat_f, rho_e, fibers, inc, target=0.100):
    """G^elas with sigma_yy = target, sigma_xx = 0 at F = I   (sec. 2.5)

    inc: G = diag(g^-1/2, g, g^-1/2), sigma - p I with p = sigma_xx.
    """
    sig_fib = sum((s.rho_0 * s.mat_f.sigma(s.G) for s in fibers), jnp.zeros((3, 3)))
    G_of = (lambda x: G_matrix(x[0], 1.0 / jnp.sqrt(x[0]))) if inc else (lambda x: G_matrix(x[0], x[1]))

    def r(x):
        s = rho_e * mat_f.sigma(G_of(x)) + sig_fib
        if inc:
            return jnp.array([s[DRIVEN, DRIVEN] - s[FREE, FREE] - target])
        return jnp.array([s[DRIVEN, DRIVEN] - target, s[FREE, FREE]])

    x = jnp.array([1.1]) if inc else jnp.array([1.1, 1.0])
    for _ in range(50):
        x = x - jnp.linalg.solve(jax.jacfwd(r)(x), r(x))
    return G_of(x)


def maes_specs(model, case):
    """Constituent specs of Maes model A-E for case U/S/F (U: G = I)."""
    m = MODELS[model]
    fibers = []
    if m["fibers"]:
        for M in FIBER_DIRS:
            G = jnp.eye(3) if case == "U" else G_fiber(FIBER["g"], M)
            fibers.append(Spec(*fung(FIBER["k1"], FIBER["k2"], M), G, FIBER["rho_0"], m["k_fiber"]))
    mat = m["matrix"]
    mat_f, mat_h = neo_hookean(mat["C10"], None if m["inc"] else mat["K"])
    G_e = jnp.eye(3) if case == "U" else prestress_G(mat_f, mat["rho_0"], fibers, m["inc"])
    return [Spec(mat_f, mat_h, G_e, mat["rho_0"], m["k_matrix"], m["matrix_remodels"])] + fibers
