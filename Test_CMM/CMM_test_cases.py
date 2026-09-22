"""
U / S / F load cases run against FCMM.py (the full constrained mixture model).

Three things in CMM_test_cases.py do not match FCMM.py and are fixed here:
  1. constituent(...) has NO `growth` kwarg. Growth is a params flag: params(..., grows=).
  2. F_g_calc does not exist. FCMM.py defines F_g_iso_calc(mix) / F_g_aniso_calc(mix, ag).
     (FCMM.CMM_sigma_calc also calls the missing F_g_calc, so it is dead code as shipped.)
  3. test_single_step's signature has a non-default arg after defaulted ones -> SyntaxError.

Boundary conditions (paper axis -> index here):
  paper Z (S3 & S5 constrained)      -> axis 0, CONFINED: lam = 1 always
  paper Y (S6 fixed, S4 driven)      -> axis 1, DRIVEN
  paper X (S1 fixed, far face free)  -> axis 2, FREE: sigma_22 = 0
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

jax.config.update("jax_enable_x64", True)

from FCMM import (
    NeoHookean, Fung, params, constituent, mixture,
    sigma_solver, commit, J_g_calc, F_g_iso_calc, F_g_aniso_calc,
    window_for,
)

CONFINED_AXIS, DRIVEN_AXIS, FREE_AXIS = 0, 1, 2


def F_g_calc(mix):
    """The name FCMM.py's own CMM_sigma_calc calls but never defines."""
    return F_g_iso_calc(mix)


# ==========================================
# Builders (corrected against the real FCMM API)
# ==========================================


def build_elastin_matrix(n_max, ds, g=1.1, T=101.0, k_minus=0.0, k_plus=0.1,
                         C10=0.305, K=6.1, rho_0=1.0, phi_0=1.0,
                         grows=True, axial=DRIVEN_AXIS, isochoric=True):
    """isochoric=True reproduces FCMM.py's own prestress_stress_snapshot convention:
    G = diag(1/sqrt(g), g, 1/sqrt(g)) with the deposition stretch on the LOADED
    axis. isochoric=False is CMM_test_cases.py's diag(g,1,1), which puts the
    prestretch on the confined axis and has det(G) = g != 1."""
    material = NeoHookean(C10=C10, K=K)
    if isochoric:
        d = [1.0 / jnp.sqrt(g)] * 3
        d[axial] = g
        G = jnp.diag(jnp.array(d))
    else:
        G = jnp.diag(jnp.array([g, 1.0, 1.0]))
    par = params(material=material, T=T, G=G, k_minus=k_minus,
                 k_plus=k_plus, phi_0=phi_0, grows=grows)
    sigma_f_0 = material.sigma_f(material.sigma(G))   # per unit mass
    return constituent.allocate(par, rho_0, sigma_f_0, n_max, ds)


def build_fiber(M, n_max, ds, k1=0.0289, k2=1.23, g=1.1, T=101.0, k_minus=0.0,
                k_plus=0.1, rho_0=1.0, phi_0=1.0, grows=True):
    M = jnp.asarray(M)
    M = M / jnp.linalg.norm(M)
    material = Fung(k1, k2, M)
    P = jnp.outer(M, M)
    G = g * P + (1.0 / jnp.sqrt(g)) * (jnp.eye(3) - P)
    par = params(material=material, T=T, G=G, k_minus=k_minus,
                 k_plus=k_plus, phi_0=phi_0, grows=grows)
    sigma_f_0 = material.sigma_f(material.sigma(G))   # per unit mass
    return constituent.allocate(par, rho_0, sigma_f_0, n_max, ds)


def mix_B(n_max, ds=1.0, F0=None):
    """matrix only, no growth"""
    F0 = jnp.eye(3) if F0 is None else F0
    return mixture.allocate([build_elastin_matrix(n_max, ds, grows=False)],
                            F0, ds, n_max)


def mix_C(n_max, ds=1.0, F0=None):
    """matrix only, with growth"""
    F0 = jnp.eye(3) if F0 is None else F0
    return mixture.allocate([build_elastin_matrix(n_max, ds, grows=True)],
                            F0, ds, n_max)


def _five(grows, n_max, ds, F0):
    F0 = jnp.eye(3) if F0 is None else F0
    a = jnp.pi / 8
    cs = [
        build_elastin_matrix(n_max, ds, rho_0=0.8, phi_0=0.8, grows=grows),
        # fibers laid out about the DRIVEN axis (y): axial, transverse, +/-alpha
        build_fiber([0, 1, 0], n_max, ds, rho_0=0.05, phi_0=0.05, grows=grows),
        build_fiber([1, 0, 0], n_max, ds, rho_0=0.05, phi_0=0.05, grows=grows),
        build_fiber([jnp.sin(a), jnp.cos(a), 0], n_max, ds, rho_0=0.05, phi_0=0.05, grows=grows),
        build_fiber([-jnp.sin(a), jnp.cos(a), 0], n_max, ds, rho_0=0.05, phi_0=0.05, grows=grows),
    ]
    return mixture.allocate(cs, F0, ds, n_max)


def mix_D(n_max, ds=1.0, F0=None):
    return _five(False, n_max, ds, F0)


def mix_E(n_max, ds=1.0, F0=None):
    return _five(True, n_max, ds, F0)


# ==========================================
# Burn-in
#   sigma_j_calc evaluates the cohort integral from s=0, but the analytic
#   homeostatic value assumes an equilibrated past (integral from -inf). With
#   an empty history the first step returns only ~ds/T of the true value
#   (measured: ratio 0.0094 vs ds/T=0.0099). Running at F=I with the gains
#   ZEROED lets the integral fill without mass feedback corrupting it; it
#   converges to the snapshot sigma_f_0 (measured ratio 0.9998 after ~7T).
#   Burning in with gains ON does NOT work: sigma_f < sigma_f_0 drives
#   rho down via rate=(k_plus-k_minus)/T and the state drifts away.
# ==========================================


def burn_in_steps(T, n_T=7.0, ds=10.0):
    """How many slots burn-in will consume -- callers need this to size n_max."""
    return int(n_T * float(T) / ds)


def burn_in(mix, n_T=7.0):
    """Fill the cohort history at F=I with the gains off, then restore them.

    Functional: returns a NEW mixture (FCMM state is an immutable pytree now).
    Runs at the mixture's own ds, so size n_max with burn_in_steps(T, n_T, ds).

    Note: `grows=False` does NOT freeze rho -- it only changes Fg/Jg handling
    inside sigma_j_calc. rho still evolves through rho_calc_from_sigma_f, which
    is why the gains have to be zeroed explicitly here.
    """
    saved = [c.params for c in mix.constituents]
    off = jnp.asarray(0.0)
    mix = mix.replace(constituents=[
        constituent(c.params.with_gains(off, off), c.history) for c in mix.constituents])

    eye = jnp.eye(3)
    for _ in range(burn_in_steps(saved[0].T, n_T, float(mix.ds))):
        _, aux = sigma_solver(mix, eye)
        mix = commit(mix, eye, aux)

    return mix.replace(constituents=[
        constituent(p, c.history) for p, c in zip(saved, mix.constituents)])


# ==========================================
# Per-axis BC spec
# ==========================================


@dataclass(frozen=True)
class AxisBC:
    mode: str                    # prescribed | free | stress | force
    value: Optional[float] = None


def prescribed(v): return AxisBC("prescribed", float(v))
def free():        return AxisBC("free")
def stress(t):     return AxisBC("stress", float(t))
def force(t):      return AxisBC("force", float(t))


def sigma_and_P(lams, mix):
    F_s = jnp.diag(lams)
    sigma_total, aux = sigma_solver(mix, F_s)
    J = lams[0] * lams[1] * lams[2]
    P_diag = J / lams * jnp.diag(sigma_total)          # P_ii = J/lam_i * sigma_ii
    return sigma_total, P_diag, (F_s, sigma_total, aux)


def make_residual(axes, mix):
    unknown = [i for i, a in enumerate(axes) if a.mode != "prescribed"]

    def lams_from_x(x):
        vals, xi = [], 0
        for a in axes:
            if a.mode == "prescribed":
                vals.append(a.value)
            else:
                vals.append(x[xi]); xi += 1
        return jnp.array(vals, dtype=jnp.float64)

    def R(x):
        sigma, P, aux = sigma_and_P(lams_from_x(x), mix)
        eqs = []
        for i in unknown:
            a = axes[i]
            if a.mode == "free":
                eqs.append(sigma[i, i])
            elif a.mode == "stress":
                eqs.append(sigma[i, i] - a.value)
            elif a.mode == "force":
                eqs.append(P[i] - a.value)
        return jnp.array(eqs), aux

    return R, unknown


def newton_solve(residual, x0, tol=1e-10, max_iter=60, fd_eps=1e-7):
    x = jnp.asarray(x0, dtype=jnp.float64)
    n = x.shape[0]
    r, aux = residual(x)
    for it in range(max_iter):
        if float(jnp.max(jnp.abs(r))) < tol:
            return x, aux
        J = jnp.zeros((n, n))
        for j in range(n):
            h = fd_eps * max(1.0, abs(float(x[j])))
            rp, _ = residual(x.at[j].add(h))
            J = J.at[:, j].set((rp - r) / h)
        dx = jnp.linalg.solve(J, -r)
        if not bool(jnp.all(jnp.isfinite(dx))):
            raise RuntimeError(f"non-finite Newton step at iter {it}")
        x = x + dx
        r, aux = residual(x)
    raise RuntimeError(f"Newton not converged: |r|={float(jnp.max(jnp.abs(r))):.3e}")


# ==========================================
# Generic driver -- U/S/F are just different axes tuples
# ==========================================


def run_case(builder, axes, n_steps=40, ds=1.0, x0=None, label="", verbose=True,
             burn=True, n_max=None, T_hint=101.0):
    """n_max is the retained-cohort WINDOW, not a budget for the whole run --
    FCMM rolls the oldest cohort out once the window is full, so it no longer
    has to grow with n_steps. Passed to the builder, which allocates up front."""
    if n_max is None:
        n_max = window_for(T_hint, ds) + 2
    mix = builder(n_max, ds=ds)
    if burn:
        mix = burn_in(mix)
    sigma_f_0 = [float(c.history.sigma_f[0]) for c in mix.constituents]

    if x0 is None:
        x0 = jnp.ones(sum(1 for a in axes if a.mode != "prescribed"))
    x = jnp.asarray(x0, dtype=jnp.float64)

    lam_h, sig_h, P_h, sf_h, s_h = [], [], [], [], []
    for _ in range(n_steps + 1):
        R, _ = make_residual(axes, mix)
        x, aux = newton_solve(R, x)
        F_s, sigma_total, trial_aux = aux
        mix = commit(mix, F_s, trial_aux)

        lams = jnp.diag(F_s)
        Pd = (lams[0] * lams[1] * lams[2]) / lams * jnp.diag(sigma_total)
        s_h.append(float(mix.history.n) * ds)
        lam_h.append([float(v) for v in lams])
        sig_h.append([float(v) for v in jnp.diag(sigma_total)])
        P_h.append([float(v) for v in Pd])
        sf_h.append([float(c.history.sigma_f[c.history.n]) for c in mix.constituents])

    n_c = len(sigma_f_0)
    dev0 = [abs(sf_h[0][j] - sigma_f_0[j]) for j in range(n_c)]
    dev1 = [abs(sf_h[-1][j] - sigma_f_0[j]) for j in range(n_c)]
    relaxing = all(dev1[j] <= dev0[j] + 1e-12 for j in range(n_c))
    held = {}
    for i, a in enumerate(axes):
        if a.mode in ("stress", "force"):
            ser = [row[i] for row in (sig_h if a.mode == "stress" else P_h)]
            held[i] = all(abs(v - a.value) < 1e-6 * max(1.0, abs(a.value)) for v in ser)
    finite = bool(jnp.all(jnp.isfinite(jnp.asarray(sig_h))))

    if verbose:
        print(f"  case {label}")
        print(f"    lam  : [{lam_h[0][0]:.5f} {lam_h[0][1]:.5f} {lam_h[0][2]:.5f}]"
              f" -> [{lam_h[-1][0]:.5f} {lam_h[-1][1]:.5f} {lam_h[-1][2]:.5f}]")
        print(f"    sigma_yy: {sig_h[0][1]:+.6e} -> {sig_h[-1][1]:+.6e}")
        print(f"    P_yy    : {P_h[0][1]:+.6e} -> {P_h[-1][1]:+.6e}")
        for j in range(n_c):
            print(f"    c{j}: sigma_f_0={sigma_f_0[j]:+.4e} "
                  f"dev {dev0[j]:.4e} -> {dev1[j]:.4e}")
        if held:
            print(f"    target held: {held}")
        print(f"    relaxing={relaxing}  finite={finite}")
    return dict(s=s_h, lam=lam_h, sigma=sig_h, P=P_h, sigma_f=sf_h,
                held=held, relaxing=relaxing, finite=finite)


def reference_state(builder, ds=1.0):
    axes = (prescribed(1.0), prescribed(1.0), free())
    out = run_case(builder, axes, n_steps=0, ds=ds, label="ref", verbose=False, burn=True)
    return out["lam"][0][FREE_AXIS], out["sigma"][0][DRIVEN_AXIS], out["P"][0][DRIVEN_AXIS]


def case_U(builder, lam_y=1.2, **kw):
    return run_case(builder, (prescribed(1.0), prescribed(lam_y), free()),
                    label="U", **kw)


def case_S(builder, increase=0.20, ds=1.0, **kw):
    _, s0, _ = reference_state(builder, ds=ds)
    return run_case(builder, (prescribed(1.0), stress(s0 * (1 + increase)), free()),
                    ds=ds, x0=[1.0, 1.0], label="S", **kw)


def case_F(builder, increase=0.20, ds=1.0, **kw):
    _, _, P0 = reference_state(builder, ds=ds)
    return run_case(builder, (prescribed(1.0), force(P0 * (1 + increase)), free()),
                    ds=ds, x0=[1.0, 1.0], label="F", **kw)


BUILDERS = {"B matrix/no-growth": mix_B, "C matrix/growth": mix_C,
            "D 5-const/no-growth": mix_D, "E 5-const/growth": mix_E}


if __name__ == "__main__":
    for name, b in BUILDERS.items():
        print("=" * 60); print(name); print("=" * 60)
        for fn in (case_U, case_S, case_F):
            try:
                fn(b, n_steps=30)
            except Exception as e:
                print(f"  {fn.__name__}: FAILED {type(e).__name__}: {e}")
            print()