"""Code verification of FCMM and HCMM against closed-form solutions at constant F, k = 0.

After a jump F: I -> F at t = 0 (history at homeostasis, rho const):
    sigma_CMM(t) = rho/J [e^(-t/T) sigma(F G) + (1 - e^(-t/T)) sigma(G)]   (Eq. 4, 5, 8 exact)
HCMM in the linear regime (Cyron 2016, Eq. 22):
    sigma(t) - sigma_pre = (sigma(0+) - sigma_pre) e^(-t/T)
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import FCMM
import HCMM
from maes_sweep import burn_in

T, DAYS, DRIVEN = 101.0, 1000.0, 1
C10, K = 0.305, 6.10
FIB = dict(k1=0.578, k2=1.23, M=(0.0, 1.0, 0.0))


def G_matrix(g):
    """G = diag(g^-1/2, g, g^-1/2)"""
    return jnp.diag(jnp.array([1 / jnp.sqrt(g), g, 1 / jnp.sqrt(g)]))


def G_fiber(g, M):
    """G = g M(x)M + g^-1/2 (I - M(x)M)"""
    M = jnp.asarray(M)
    P = jnp.outer(M, M)
    return g * P + (1 / jnp.sqrt(g)) * (jnp.eye(3) - P)


def materials(kind):
    if kind == "matrix":
        return FCMM.NeoHookean(C10, K), HCMM.NeoHookean(C10, K)
    M = jnp.asarray(FIB["M"])
    return FCMM.Fung(FIB["k1"], FIB["k2"], M), HCMM.Fung(FIB["k1"], FIB["k2"], M)


def exact(mat, F, G, t):
    """sigma_CMM(t) for rho = 1"""
    J = jnp.linalg.det(F)
    w = jnp.exp(-t / T)[:, None, None]
    return (w * mat.sigma(F @ G) + (1 - w) * mat.sigma(G)) / J


def run_fcmm(mat, G, F, ds):
    n_max = FCMM.window_for(T, ds) + 2
    par = FCMM.params(material=mat, T=T, G=G, k_minus=0.0, k_plus=0.0, phi_0=1.0)
    c = FCMM.constituent.allocate(par, 1.0, mat.sigma_f(mat.sigma(G)), n_max, ds)
    mix = burn_in(FCMM.mixture.allocate([c], jnp.eye(3), ds, n_max), n_max)
    out = []
    for _ in range(int(DAYS / ds)):
        s, aux = FCMM.sigma_solver(mix, F)
        mix = FCMM.commit(mix, F, aux)
        out.append(s)
    return jnp.stack(out), jnp.arange(1, len(out) + 1) * ds


def run_hcmm(mat, G, F, ds):
    c = HCMM.constituent(mat, T=T, rho_0=1.0, k_sigma_plus=0.0, k_sigma_minus=0.0, G=G)
    mix = HCMM.mixture([c], ds=ds)
    out = []
    for _ in range(int(DAYS / ds)):
        s, aux = HCMM.sigma_solver(mix, F)
        mix = HCMM.commit(mix, F, aux)
        out.append(s)
    return jnp.stack(out), jnp.arange(len(out)) * ds


def err(sig, ref):
    """max_t |sigma_yy - ref_yy| / |ref_yy(0+) - ref_yy(inf)|"""
    scale = jnp.abs(ref[0, DRIVEN, DRIVEN] - ref[-1, DRIVEN, DRIVEN])
    scale = jnp.where(scale < 1e-12, jnp.abs(ref[0, DRIVEN, DRIVEN]), scale)
    return float(jnp.max(jnp.abs(sig[:, DRIVEN, DRIVEN] - ref[:, DRIVEN, DRIVEN])) / scale)


def homeostasis_drift(kind, g, ds=10.0):
    """max_t |sigma(t) - sigma(0)| at F = I"""
    mf, mh = materials(kind)
    G = G_matrix(g) if kind == "matrix" else G_fiber(g, FIB["M"])
    sf, _ = run_fcmm(mf, G, jnp.eye(3), ds)
    sh, _ = run_hcmm(mh, G, jnp.eye(3), ds)
    return (float(jnp.max(jnp.abs(sf - sf[0]))), float(jnp.max(jnp.abs(sh - sh[0]))))


if __name__ == "__main__":
    print("1) homeostasis at F = I, 1000 days: max |sigma(t) - sigma(0)|  (FCMM / HCMM)")
    for kind, g in (("matrix", 1.2), ("fiber", 1.1)):
        d = homeostasis_drift(kind, g)
        print(f"   {kind:6s} g={g}: {d[0]:.2e} / {d[1]:.2e}")

    print("\n2) constant F after jump, error vs exact CMM, relative to the relaxation amplitude")
    cases = [("matrix", 1.0, 1.2), ("matrix", 1.2, 1.2), ("fiber", 1.1, 1.2),
             ("matrix", 1.2, 1.001), ("fiber", 1.1, 1.001)]
    for kind, g, lam in cases:
        mf, mh = materials(kind)
        G = G_matrix(g) if kind == "matrix" else G_fiber(g, FIB["M"])
        F = jnp.diag(jnp.array([1.0, lam, 1.0 / lam]))
        row = []
        for ds in (10.0, 5.0, 2.5):
            sf, tf = run_fcmm(mf, G, F, ds)
            sh, th = run_hcmm(mh, G, F, ds)
            row.append((err(sf, exact(mf, F, G, tf)), err(sh, exact(mf, F, G, th))))
        print(f"   {kind:6s} g={g} lam={lam}:  " + "  ".join(
            f"ds={ds:4.1f} F {100*a:5.2f}% H {100*b:5.2f}%"
            for ds, (a, b) in zip((10.0, 5.0, 2.5), row)))
