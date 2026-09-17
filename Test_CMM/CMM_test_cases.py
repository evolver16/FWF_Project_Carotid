import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from CMM import *

# ==========================================
# Shared setup
#   Single isotropic NeoHookean matrix constituent, uniaxial strain:
#   F = diag(lambda, 1, 1). J != 1 in general (compressible material), which
#   matters: trace(sigma) is identically zero at J=1 for this compressible
#   NeoHookean model (deviatoric part is trace-free, volumetric part vanishes
#   at J=1), so an incompressible reduction would make sigma_f trivially zero
#   everywhere and silently break every stress-dependent remodeling law.
# ==========================================


def build_F(lam):
    return jnp.diag(jnp.array([lam, 1.0, 1.0]))

def build_elastin_matrix(g=1.1, T=101.0, k_minus=0.0, k_plus=0.1, C10=0.305, K=6.1, rho_0=1.0, phi_0=1.0, growth=True):
    material = NeoHookean(C10=C10, K=K)
    G = jnp.diag(jnp.array([g, 1.0, 1.0]))
    par = params(material=material, T=T, G=G, k_minus=k_minus, k_plus=k_plus, phi_0=phi_0)
    sigma_f_0 = (rho_0 / phi_0) * material.sigma_f(material.sigma(G))
    return constituent(par, rho_0=rho_0, sigma_f_0=sigma_f_0, growth=growth)


def build_fiber(M, k1=0.0289, k2=1.23, g=1.1, T=101.0, k_minus=0.0, k_plus=0.1, rho_0=1.0, phi_0=1.0, growth=True):
    M = jnp.asarray(M, dtype=jnp.float64)
    M = M / jnp.linalg.norm(M)
    material = Fung(k1, k2, M)

    P = jnp.outer(M, M)
    G = g * P + (1.0 / jnp.sqrt(g)) * (jnp.eye(3) - P)

    par = params(material=material, T=T, G=G, k_minus=k_minus, k_plus=k_plus, phi_0=phi_0)
    sigma_f_0 = (rho_0 / phi_0) * material.sigma_f(material.sigma(G))
    return constituent(par, rho_0=rho_0, sigma_f_0=sigma_f_0, growth=growth)


def test_A_mixture(F0=jnp.eye(3),ds=1.0):
    incomp_elastin_matrix = build_elastin_matrix(K=6e10,growth=False)
    return mixture([incomp_elastin_matrix],F0,ds=ds)

def test_B_mixture(F0=jnp.eye(3),ds=1.0):
    incomp_elastin_matrix = build_elastin_matrix(growth=False)
    return mixture([incomp_elastin_matrix],F0,ds=ds)

def test_C_mixture(F0=jnp.eye(3),ds=1.0):
    incomp_elastin_matrix = build_elastin_matrix()
    return mixture([incomp_elastin_matrix],F0,ds=ds)

def test_D_mixture(F0=jnp.eye(3),ds=1.0):
    incomp_elastin_matrix = build_elastin_matrix(K=6e10, rho_0=0.8, phi_0=0.8, growth=False)
    alpha = jnp.pi/8
    fiber_x = build_fiber(M=[1,0,0], rho_0=0.05, phi_0=0.05, growth=False)
    fiber_y = build_fiber(M=[0,1,0], rho_0=0.05, phi_0=0.05, growth=False)
    fiber_alpha1 = build_fiber(M=[jnp.sin(alpha),jnp.cos(alpha),0], rho_0=0.05, phi_0=0.05, growth=False)
    fiber_alpha2 = build_fiber(M=[-jnp.sin(alpha),jnp.cos(alpha),0], rho_0=0.05, phi_0=0.05, growth=False)
    return mixture([incomp_elastin_matrix,fiber_x,fiber_y,fiber_alpha1,fiber_alpha2],F0,ds=ds)

def test_E_mixture(F0=jnp.eye(3),ds=1.0):
    incomp_elastin_matrix = build_elastin_matrix(K=6e10, rho_0=0.8, phi_0=0.8, growth=False)
    alpha = jnp.pi/8
    fiber_x = build_fiber(M=[1,0,0], rho_0=0.05, phi_0=0.05, growth=True)
    fiber_y = build_fiber(M=[0,1,0], rho_0=0.05, phi_0=0.05, growth=True)
    fiber_alpha1 = build_fiber(M=[jnp.sin(alpha),jnp.cos(alpha),0], rho_0=0.05, phi_0=0.05, growth=True)
    fiber_alpha2 = build_fiber(M=[-jnp.sin(alpha),jnp.cos(alpha),0], rho_0=0.05, phi_0=0.05, growth=True)
    return mixture([incomp_elastin_matrix,fiber_x,fiber_y,fiber_alpha1,fiber_alpha2],F0,ds=ds)

def sigma11_of_lambda(lam, mix, J_g):
    Fg_s = F_g_calc(mix)
    F_s = build_F(lam)
    sigma_total, results, mix_hist_trial = evaluate_trial(F_s, Fg_s, mix, J_g)
    return sigma_total[0, 0], (F_s, Fg_s, sigma_total, results, mix_hist_trial)


def bisect_lambda_for_stress(target_sigma11, mix, J_g, lo=0.2, hi=20.0, iters=60):
    """sigma_11(lambda) is monotonically increasing, but as the material remodels
    its own residual (lambda=1) stress can drift above the target -- holding a
    fixed target may then require lambda < 1 (compression), so the bracket must
    allow that, and a bracket failure must be surfaced, not silently returned as
    the edge of the search range."""
    f_lo, _ = sigma11_of_lambda(jnp.asarray(lo), mix, J_g)
    f_hi, _ = sigma11_of_lambda(jnp.asarray(hi), mix, J_g)
    if (f_lo - target_sigma11) * (f_hi - target_sigma11) > 0:
        raise RuntimeError(
            f"target sigma_11={target_sigma11:.6f} not bracketed: "
            f"sigma_11({lo})={float(f_lo):.6f}, sigma_11({hi})={float(f_hi):.6f}"
        )
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid, aux = sigma11_of_lambda(jnp.asarray(mid), mix, J_g)
        if (f_mid - target_sigma11) * (f_lo - target_sigma11) <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return mid, aux


# ==========================================
# Test 1: constant elongation (displacement-controlled, case U)
#   Stretch is applied once and held fixed; remodeling relaxes the stress.
# ==========================================


def test_constant_elongation(n_steps=60, lam_target=1.2):
    """Stress remodeling is governed by sigma_f = trace(sigma), not any single
    tensor component, so relaxation is checked on sigma_f (equivalently
    c.history.sigma_f), toward the homeostatic reference sigma_f_0."""
    mix, c, par = build_mixture()
    J_g = J_g_calc(mix)
    sigma_f_0 = float(c.history.sigma_f[0])

    F_s = build_F(jnp.asarray(lam_target))
    Fg_s = F_g_calc(mix)
    sigma_total, results, mix_hist_trial = evaluate_trial(F_s, Fg_s, mix, J_g)
    commit_step(mix, mix_hist_trial, results)

    s_vals = [0.0, float(mix.history.s[-1])]
    sigma11_vals = [sigma_f_0, float(sigma_total[0, 0])]
    sigma_f_vals = [sigma_f_0, float(c.history.sigma_f[-1])]

    for _ in range(n_steps):
        J_g = J_g_calc(mix)
        Fg_s = F_g_calc(mix)
        sigma_total, results, mix_hist_trial = evaluate_trial(F_s, Fg_s, mix, J_g)
        commit_step(mix, mix_hist_trial, results)
        s_vals.append(float(mix.history.s[-1]))
        sigma11_vals.append(float(sigma_total[0, 0]))
        sigma_f_vals.append(float(c.history.sigma_f[-1]))

    deviation = [abs(sf - sigma_f_0) for sf in sigma_f_vals]
    relaxing = deviation[-1] < deviation[1]
    all_finite = all(jnp.isfinite(jnp.asarray(sigma_f_vals)))

    print("=== Test 1: constant elongation (case U) ===")
    print(f"stretch held at lambda = {lam_target}")
    print(f"sigma_f_0 (homeostatic reference) = {sigma_f_0:.6f}")
    print(
        f"sigma_f, right after jump = {sigma_f_vals[1]:.6f}  (deviation {deviation[1]:.6f})"
    )
    print(
        f"sigma_f, end of run      = {sigma_f_vals[-1]:.6f}  (deviation {deviation[-1]:.6f})"
    )
    print(f"deviation from homeostasis shrinking: {relaxing}")
    print(f"all finite: {all_finite}")
    print()
    return s_vals, sigma11_vals, sigma_f_vals


# ==========================================
# Test 2: constant stress (stress-controlled, case S)
#   Stress is stepped once and held fixed via an outer bisection on lambda
#   at every time step; remodeling creeps the stretch upward.
# ==========================================


def test_constant_stress(n_steps=60, stress_increase=0.20):
    """Target is defined on sigma_11 itself (the controlled component), evaluated
    at the undeformed F=I state, not on sigma_f (trace) -- comparing a component
    target against a trace-based reference silently makes the target unreachable."""
    mix, c, par = build_mixture()
    J_g = J_g_calc(mix)
    sigma11_ref, _ = sigma11_of_lambda(jnp.asarray(1.0), mix, J_g)
    target_sigma11 = float(sigma11_ref) * (1.0 + stress_increase)

    lam, aux = bisect_lambda_for_stress(target_sigma11, mix, J_g)
    F_s, Fg_s, sigma_total, results, mix_hist_trial = aux
    commit_step(mix, mix_hist_trial, results)

    s_vals = [0.0, float(mix.history.s[-1])]
    lam_vals = [1.0, float(lam)]
    sigma11_vals = [float(sigma11_ref), float(sigma_total[0, 0])]

    for _ in range(n_steps):
        J_g = J_g_calc(mix)
        lam, aux = bisect_lambda_for_stress(target_sigma11, mix, J_g)
        F_s, Fg_s, sigma_total, results, mix_hist_trial = aux
        commit_step(mix, mix_hist_trial, results)
        s_vals.append(float(mix.history.s[-1]))
        lam_vals.append(float(lam))
        sigma11_vals.append(float(sigma_total[0, 0]))

    stress_held = all(abs(s - target_sigma11) < 1e-4 for s in sigma11_vals[1:])
    increasing = all(
        lam_vals[i + 1] >= lam_vals[i] - 1e-9 for i in range(len(lam_vals) - 1)
    )
    decreasing = all(
        lam_vals[i + 1] <= lam_vals[i] + 1e-9 for i in range(len(lam_vals) - 1)
    )
    monotonic = increasing or decreasing
    all_finite = all(jnp.isfinite(jnp.asarray(lam_vals)))

    # Depositing new material at the fixed prestretch G keeps adding residual
    # stress even without external stretch, so with k_sigma+ > 0 and no offsetting
    # degradation, a fixed stress target can require increasing COMPRESSION over
    # time (lambda decreasing) once residual stress alone would exceed it --
    # not necessarily the increasing stretch of the paper's own case S example.
    direction = (
        "decreasing (compression, residual stress outgrew the target)"
        if decreasing
        else "increasing" if increasing else "non-monotonic"
    )

    print("=== Test 2: constant stress (case S) ===")
    print(
        f"target sigma_11 = {target_sigma11:.6f} ({stress_increase*100:.0f}% above homeostatic)"
    )
    print(f"lambda(0)   = {lam_vals[0]:.6f}")
    print(f"lambda(end) = {lam_vals[-1]:.6f}")
    print(f"stress held at target throughout: {stress_held}")
    print(f"stretch trajectory: {direction}")
    print(f"monotonic: {monotonic}")
    print(f"all finite: {all_finite}")
    print()
    return s_vals, lam_vals, sigma11_vals


# ==========================================
# Test 3: single-step internal consistency
#   Verifies the Newton solve at one step is self-consistent: the converged
#   sigma_f equals material.sigma_f(sigma_j), and rho_calc_from_sigma_f's
#   closed form agrees with a brute-force trapezoidal integration of Eq. 36
#   evaluated on the resulting (committed) m history.
# ==========================================


def rho_rk4_reference(
    rho_prev, sigma_f_prev, sigma_f_new, sigma_f_0, par, ds, n_sub=2000
):
    """Fine RK4 integration of dot(rho) = rho*(k+ - k-)/T * sigma_frac(tau), with
    sigma_frac linearly interpolated between its two step endpoints. A valid
    ground truth at ANY step size, unlike Eq. 36's trapezoidal discretization,
    which the paper itself only claims is accurate once s >> T (Q(s) ~ 0)."""
    rate = (par.k_sigma_plus - par.k_sigma_minus) / par.T
    frac_prev = (sigma_f_prev - sigma_f_0) / sigma_f_0
    frac_new = (sigma_f_new - sigma_f_0) / sigma_f_0

    def f(t_frac, rho):
        frac = frac_prev + t_frac * (frac_new - frac_prev)
        return rate * frac * rho

    h = 1.0 / n_sub
    rho = rho_prev
    t = 0.0
    for _ in range(n_sub):
        k1 = f(t, rho)
        k2 = f(t + h / 2, rho + h / 2 * k1)
        k3 = f(t + h / 2, rho + h / 2 * k2)
        k4 = f(t + h, rho + h * k3)
        rho = rho + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return rho


def test_single_step(lam_target=1.15, ds_values=(10.0, 2.0, 0.5)):
    """The closed-form rho update is itself a trapezoidal (second-order)
    approximation of the ODE dot(rho)=rho*(k+-k-)/T*sigma_frac -- it should
    match the fine RK4 reference only approximately at large ds, with the
    error shrinking as ds shrinks. This checks convergence rather than a
    single fixed tolerance, since ds=10 alone is not close to the continuum
    limit relative to T=101 (matches the ~0.4% gap found analytically earlier
    for this same comparison)."""
    residual = None
    errors = []
    for ds in ds_values:
        mix, c, par = build_mixture(ds=ds)
        J_g = J_g_calc(mix)
        Fg_s = F_g_calc(mix)
        F_s = build_F(jnp.asarray(lam_target))
        sigma_f_0 = float(c.history.sigma_f[0])
        rho_prev = float(c.history.rho[-1])
        sigma_f_prev = float(c.history.sigma_f[-1])

        sigma_total, results, mix_hist_trial = evaluate_trial(F_s, Fg_s, mix, J_g)
        sf_s, sigma_j, rho_s, m_s, K_cumu_s = results[0]

        if residual is None:
            residual = float(par.material.sigma_f(sigma_j) - sf_s)

        commit_step(mix, mix_hist_trial, results)
        rho_closed = float(c.history.rho[-1])
        rho_rk4 = rho_rk4_reference(
            rho_prev, sigma_f_prev, float(sf_s), sigma_f_0, par, ds
        )
        rel_err = abs(rho_closed - rho_rk4) / abs(rho_rk4)
        errors.append(rel_err)

    residual_ok = abs(residual) < 1e-8
    converging = all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
    all_finite = all(jnp.isfinite(jnp.asarray(errors)))

    print("=== Test 3: single-step internal consistency ===")
    print(
        f"Newton residual (sigma_f_true - sigma_f*) at ds={ds_values[0]}: {residual:.3e}"
    )
    print("rho (closed form) vs fine RK4 reference, relative error by step size:")
    for ds, err in zip(ds_values, errors):
        print(f"  ds={ds:6.2f}: relative error = {err:.3e}")
    print(f"residual below tol: {residual_ok}")
    print(f"error shrinks as ds shrinks (converging to RK4): {converging}")
    print(f"all finite: {bool(all_finite)}")
    print()
    return residual, errors


if __name__ == "__main__":
    test_constant_elongation()
    test_constant_stress()
    test_single_step()
