"""Maes & Famaey (2023) Table 1/2 models: constituent specs, deposition stretches, prestress, FCMM / HCMM builders."""

import pathlib
from dataclasses import dataclass
from functools import partial

import jax

jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(__file__).parent / "compiled"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp

import fcmm
import hcmm
import materials
from verification import material_point

DRIVEN, FREE = material_point.DRIVEN_AXIS, material_point.FREE_AXIS
T_DAYS = 101.0


# ---------- constituent specs and deposition stretches ----------

@dataclass
class Spec:
    """One constituent: material (shared by FCMM and HCMM), deposition stretch, mass, turnover"""
    mat: object
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


# ---------- builders ----------

def burn_in(mix, n_steps):
    """Fill the FCMM cohort history at F = I with gains off."""
    saved = [c.params for c in mix.constituents]
    off = jnp.asarray(0.0)
    gains_off = lambda p: (p.replace(production=p.production.replace(k=off), removal=p.removal.replace(k=off))
                           if p.remodels else p)
    mix = mix.replace(constituents=[fcmm.constituent(gains_off(c.params), c.history)
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
        par = fcmm.params(s.mat, s.G, *(fcmm.maes(s.T, s.k_plus) if s.remodels else (None, None)))
        cs.append(fcmm.constituent.allocate(par, s.rho_0, s.mat.sigma_f(s.mat.sigma(s.G)), n_max, ds))
    mix = fcmm.mixture.allocate(cs, jnp.eye(3), ds, n_max, fcmm.Isotropic() if ag is None else fcmm.Anisotropic(ag))
    return burn_in(mix, n_max)


def build_hcmm(specs, ds, ag=None, mode="deposition"):
    cs = [hcmm.constituent(s.mat, s.rho_0, *(hcmm.maes(s.T, s.k_plus) if s.remodels else (None, None)), G=s.G,
                           sigma_pre_mode=mode)
          for s in specs]
    return hcmm.mixture(cs, ds=ds, growth=hcmm.Isotropic() if ag is None else hcmm.Anisotropic(ag))


VARIANTS = {
    "FCMM": (build_fcmm, fcmm),
    "HCMM dep": (partial(build_hcmm, mode="deposition"), hcmm),
    "HCMM init": (partial(build_hcmm, mode="initial"), hcmm),
}


# ---------- Maes & Famaey (2023) Table 1/2, code axes (Z, Y, X) ----------

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


def matrix_material(C10, K=None):
    """neo-Hookean matrix, K = None: incompressible (Eq. 32)"""
    return materials.NeoHookeanInc(C10) if K is None else materials.NeoHookean(C10, K)


def bc_for(model, case):
    return material_point.BC(case, TARGETS[case], hybrid=MODELS[model]["inc"])


def prestress_G(mat, rho_e, fibers, inc, target=0.100):
    """G^elas with sigma_yy = target, sigma_xx = 0 at F = I   (sec. 2.5)

    inc: G = diag(g^-1/2, g, g^-1/2), sigma - p I with p = sigma_xx.
    """
    sig_fib = sum((s.rho_0 * s.mat.sigma(s.G) for s in fibers), jnp.zeros((3, 3)))
    G_of = (lambda x: G_matrix(x[0], 1.0 / jnp.sqrt(x[0]))) if inc else (lambda x: G_matrix(x[0], x[1]))

    def r(x):
        s = rho_e * mat.sigma(G_of(x)) + sig_fib
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
            fiber = materials.Fung(FIBER["k1"], FIBER["k2"], jnp.asarray(M))
            fibers.append(Spec(fiber, G, FIBER["rho_0"], m["k_fiber"]))
    mat = matrix_material(m["matrix"]["C10"], None if m["inc"] else m["matrix"]["K"])
    G_e = jnp.eye(3) if case == "U" else prestress_G(mat, m["matrix"]["rho_0"], fibers, m["inc"])
    return [Spec(mat, G_e, m["matrix"]["rho_0"], m["k_matrix"], m["matrix_remodels"])] + fibers
