"""Full constrained mixture model (FCMM), Maes & Famaey (2023) sec. 2.1.

Every constituent is a superposition of cohorts deposited at times tau, each
carrying its own elastic deformation:

    rho^j(s)   = rho_0^j Q^j(s) + int_0^s m^j(tau) q^j(s,tau) dtau    (Eq. 1)
    F_e^j(s,tau) = F(s) F_g(s)^-1 [F(tau) F_g(tau)^-1]^-1 G^j         (Eq. 2/5)
    sigma^j(s) = (J_g/J)(rho_0^j/phi_0^j)
                 int_0^s m^j q^j (dW^j/dF) F^T dtau                   (Eq. 4)
    m^j(s)     = (rho^j/T^j)(1 + k_sigma_+ rel)                       (Eq. 11)
    q^j(s,tau) = exp(-int_tau^s K_-^j dtau')                          (Eq. 12)

The cohort integrals are evaluated by the trapezoid rule over retained
cohorts (Eq. 34/35). m^j(s) depends on sigma^j(s) and vice versa, so each step
closes that loop with a scalar Newton solve on sigma_f (sec. 2.4.1).

History buffers are preallocated to n_max entries and used as a ROLLING
WINDOW, so shapes are static (jittable, no retrace per step) and cost is
bounded regardless of run length. Slots 0..n are valid; unfilled slots hold
identity (F, Fg) or zero, and the trapezoid abscissa is clamped at n so they
sit in zero-width segments. F and Fg must stay finite there, since det(F)=0
would reach the integrand as nan through 0*nan.

Time is carried in step units: T is supplied in days and divided by ds once
(T_steps = T/ds), and the abscissa is the step index. Identical to carrying
days with explicit ds factors, since m scales by ds while dtau shrinks by ds.

Solver entry point (shared with HCMM.py -- see jaxFEM_solver.py):
    sigma_tot, aux = sigma_solver(state, F)   pure, committed state only
    state          = commit(state, F, aux)    once, on the converged F
"""

import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================


@jit
def F_e_calc(F, F_g, G, n):
    """F_e^j(s,tau) = F(s) F_g(s)^-1 [F(tau) F_g(tau)^-1]^-1 G^j   (Eq. 2/5)

    F, F_g are the full (n_max,3,3) buffers, so this returns one F_e per
    retained cohort; n indexes the current step.
    """
    inner = F @ jnp.linalg.inv(F_g)
    return F[n] @ jnp.linalg.inv(F_g[n]) @ jnp.linalg.inv(inner) @ G


def F_g_iso_calc(mixt):
    """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 6, isotropic)"""
    return J_g_calc(mixt) ** (1.0 / 3.0) * jnp.eye(3)


def F_g_aniso_calc(mixt, ag):
    """F_g(s) = (rho_tot(s)/rho_tot(0) - 1) ag(x)ag + I   (Eq. 7, anisotropic)"""
    return (J_g_calc(mixt) - 1) * jnp.outer(ag, ag) + jnp.eye(3)


def J_g_calc(mixt):
    """J_g(s) = det F_g(s) = rho_tot(s)/rho_tot(0)                 (Eq. 6/7)

    Lagged: taken from the last committed step.
    """
    return rho_tot_prev_calc(mixt) / mixt.rho_tot_0


def step_axis(n, n_max):
    """Abscissa tau for the cohort integrals (Eq. 34/35), in step units.

    Clamped past n so unfilled slots sit in zero-width trapezoid segments.
    Not jitted: n_max is a static shape and jnp.arange needs it concrete.
    """
    return jnp.minimum(jnp.arange(n_max), n).astype(jnp.result_type(float))


def T_steps(par, ds):
    """T^j expressed in steps: T_steps = T/ds."""
    return par.T / ds


def window_for(T, ds, tol=1e-3, per_cohort=False):
    """How many cohorts to retain before dropping the oldest.

    Default bounds the TOTAL dropped tail, sum_{k>=n} exp(-k*ds/T) < tol, i.e.
        exp(-n*ds/T) / (1 - exp(-ds/T)) < tol.

    per_cohort=True is the paper's sec. 2.6 rule, which bounds a SINGLE
    dropped cohort (exp(-n*ds/T) < tol) and gives the noTimes = 69 hard-coded
    in UMAT_GR_CM.for. Being a per-term rather than a tail bound, it is looser
    by roughly T/ds.
    """
    import math
    T, ds = float(T), float(ds)
    if per_cohort:
        return int(math.ceil(-math.log(tol) * T / ds))
    return int(math.ceil(-math.log(tol * (1.0 - math.exp(-ds / T))) * T / ds))


# ==========================================
# Material
# ==========================================


@jax.tree_util.register_pytree_node_class
class Fung:
    def __init__(self, k1, k2, M):
        self.k1 = k1
        self.k2 = k2
        M = jnp.asarray(M)
        self.M = M / jnp.linalg.norm(M)

    def tree_flatten(self):
        return (self.k1, self.k2, self.M), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def Psi(self, F):
        """W^coll(I4) = k1/(2 k2) [exp(k2 (I4 - 1)^2) - 1]            (Eq. 28)

        with I4 = |F M|^2.
        """
        FM = F @ self.M
        I4 = jnp.dot(FM, FM)
        return (self.k1 / (2 * self.k2)) * (jnp.exp(self.k2 * (I4 - 1) ** 2) - 1)

    @jit
    def sigma(self, F):
        """sigma = (dW/dF) F^T                                        (Eq. 29)

        The rho_0^j/(phi_0^j J) prefactor of Eq. 4 is applied by sigma_j_calc.
        """
        dW_dF = jax.grad(self.Psi)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = I_1(sigma) = tr(sigma)                           (Eq. 31)"""
        return jnp.trace(sigma)


@jax.tree_util.register_pytree_node_class
class NeoHookean:
    def __init__(self, C10, K):
        self.C10 = C10
        self.K = K

    def tree_flatten(self):
        return (self.C10, self.K), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def Psi(self, F):
        """W^elas = C10 (I1bar - 3) + K/2 (J_e - 1)^2                 (Eq. 28)

        with I1bar = J_e^(-2/3) tr(F^T F) and J_e = det F_e.
        """
        I1 = jnp.trace(F.T @ F)
        J = jnp.linalg.det(F)
        I1_inc = I1 * J ** (-2 / 3)
        return self.C10 * (I1_inc - 3) + self.K / 2 * (J - 1) ** 2

    @jit
    def sigma(self, F):
        """sigma = (dW/dF) F^T                                        (Eq. 29)"""
        dW_dF = jax.grad(self.Psi)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = I_1(sigma) = tr(sigma)                           (Eq. 31)"""
        return jnp.trace(sigma)


# ==========================================
# State containers -- all preallocated
# ==========================================


def _advance(buf, value, k, shift):
    """Write `value` at slot k, dropping the oldest entry first if the window
    is full. Branchless, so `shift` may be a tracer."""
    return jnp.where(shift, jnp.roll(buf, -1, axis=0), buf).at[k].set(value)


@jax.tree_util.register_pytree_node_class
class history:
    """Per-cohort state of one constituent: deposition rate m^j(tau) (Eq. 11),
    scalar stress sigma_f^j(tau) (Eq. 31), cumulative degradation K_cumu(tau)
    (Eq. 12) and density rho^j(tau) (Eq. 1).

    Fixed length n_max, used as a rolling window: once full, the oldest cohort
    is dropped, as UMAT_GR_CM.for does over its noTimes = 69 slots.

    sigma_f_0 and rho_0 sit outside the buffers -- they are the homeostatic
    setpoint sigma_f^j(0) and reference density rho^j(0), which a shift would
    otherwise scroll out of slot 0.
    """

    def __init__(self, m, sigma_f, K_cumu, rho, n, sigma_f_0, rho_0):
        self.m = m
        self.sigma_f = sigma_f
        self.K_cumu = K_cumu
        self.rho = rho
        self.n = n
        self.sigma_f_0 = sigma_f_0
        self.rho_0 = rho_0

    @classmethod
    def allocate(cls, n_max, m_0, sigma_f_0, rho_0):
        z = jnp.zeros(n_max)
        return cls(
            m=z.at[0].set(m_0),
            sigma_f=z.at[0].set(sigma_f_0),
            K_cumu=z,                       # K_cumu(0) = 0
            rho=z.at[0].set(rho_0),
            n=jnp.asarray(0),
            sigma_f_0=jnp.asarray(sigma_f_0),
            rho_0=jnp.asarray(rho_0),
        )

    @property
    def n_max(self):
        return self.m.shape[0]

    def roll_if_full(self):
        """Shift the window if full, without writing the new cohort, whose
        values the inner Newton solve has yet to produce.

        n becomes the slot of the most recent committed cohort in the new
        indexing, so the trial slot is n+1 either way. Must stay in lockstep
        with mixture_history.advance, or cohort tau sits at different slots in
        the two buffers.
        """
        shift = (self.n + 1) >= self.n_max
        roll = lambda b: jnp.where(shift, jnp.roll(b, -1, axis=0), b)
        return history(
            m=roll(self.m), sigma_f=roll(self.sigma_f),
            K_cumu=roll(self.K_cumu), rho=roll(self.rho),
            n=jnp.where(shift, self.n - 1, self.n),
            sigma_f_0=self.sigma_f_0, rho_0=self.rho_0,
        )

    def write(self, k, m_s, sigma_f_s, K_cumu_s, rho_s):
        """Write the settled cohort into slot k of an already-rolled history."""
        return history(
            m=self.m.at[k].set(m_s),
            sigma_f=self.sigma_f.at[k].set(sigma_f_s),
            K_cumu=self.K_cumu.at[k].set(K_cumu_s),
            rho=self.rho.at[k].set(rho_s),
            n=k,
            sigma_f_0=self.sigma_f_0,
            rho_0=self.rho_0,
        )

    def tree_flatten(self):
        return (self.m, self.sigma_f, self.K_cumu, self.rho, self.n,
                self.sigma_f_0, self.rho_0), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.m, obj.sigma_f, obj.K_cumu, obj.rho, obj.n,
         obj.sigma_f_0, obj.rho_0) = children
        return obj


@jax.tree_util.register_pytree_node_class
class params:
    def __init__(self, material, T, G, k_minus, k_plus, phi_0, grows=True):
        self.material = material
        self.T = jnp.asarray(T)
        self.G = jnp.asarray(G)
        self.k_sigma_minus = jnp.asarray(k_minus)
        self.k_sigma_plus = jnp.asarray(k_plus)
        self.phi_0 = jnp.asarray(phi_0)
        self.grows = grows

    def with_gains(self, k_plus, k_minus):
        """Copy with k_sigma_+ and k_sigma_- replaced."""
        return params(self.material, self.T, self.G, k_minus, k_plus,
                      self.phi_0, self.grows)

    def tree_flatten(self):
        return (self.material, self.T, self.G, self.k_sigma_minus,
                self.k_sigma_plus, self.phi_0), self.grows

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.material, obj.T, obj.G, obj.k_sigma_minus,
         obj.k_sigma_plus, obj.phi_0) = children
        obj.grows = aux_data
        return obj


@jax.tree_util.register_pytree_node_class
class constituent:
    def __init__(self, params, history):
        self.params = params
        self.history = history

    @classmethod
    def allocate(cls, params, rho_0, sigma_f_0, n_max, ds):
        """Initial cohort: m^j(0) = rho^j(0)/T^j, the homeostatic deposition
        rate per step (Eq. 10/21)."""
        m_0 = rho_0 / T_steps(params, ds)
        return cls(params, history.allocate(n_max, m_0, sigma_f_0, rho_0))

    def tree_flatten(self):
        return (self.params, self.history), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.params, obj.history = children
        return obj


@jax.tree_util.register_pytree_node_class
class mixture_history:
    """F(tau) and F_g(tau) per retained cohort, for Eq. 2/5.

    Unfilled slots hold identity so det(F) stays finite; the clamped abscissa
    keeps them out of the integrals.
    """

    def __init__(self, F, Fg, n):
        self.F = F
        self.Fg = Fg
        self.n = n

    @classmethod
    def allocate(cls, n_max, F0):
        eye = jnp.broadcast_to(jnp.eye(3), (n_max, 3, 3))
        return cls(F=eye.at[0].set(jnp.asarray(F0)), Fg=eye, n=jnp.asarray(0))

    @property
    def n_max(self):
        return self.F.shape[0]

    def advance(self, F_s, Fg_s):
        """Append the current step's F and F_g, rolling if the window is full.
        Same n and n_max as each constituent's history, so both drop the same
        cohort."""
        shift = (self.n + 1) >= self.n_max
        k = jnp.minimum(self.n + 1, self.n_max - 1)
        return mixture_history(F=_advance(self.F, F_s, k, shift),
                               Fg=_advance(self.Fg, Fg_s, k, shift), n=k)

    def tree_flatten(self):
        return (self.F, self.Fg, self.n), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.F, obj.Fg, obj.n = children
        return obj


@jax.tree_util.register_pytree_node_class
class mixture:
    def __init__(self, constituents, ds, history, rho_tot_0, ag=None):
        self.constituents = constituents
        self.ds = ds
        self.history = history
        self.rho_tot_0 = rho_tot_0
        self.ag = ag

    @classmethod
    def allocate(cls, constituents, F0, ds, n_max, ag=None):
        """n_max is the retained-cohort window; see window_for."""
        ag = None if ag is None else jnp.asarray(ag) / jnp.linalg.norm(jnp.asarray(ag))
        return cls(constituents, ds, mixture_history.allocate(n_max, F0),
                   sum(c.history.rho_0 for c in constituents), ag)

    def F_g_calc(self):
        """Mirrors HCMM.mixture.F_g_calc so sigma_solver needs no extra argument."""
        if self.ag is None:
            return F_g_iso_calc(self)
        return F_g_aniso_calc(self, self.ag)

    def replace(self, constituents=None, history=None):
        return mixture(self.constituents if constituents is None else constituents,
                       self.ds,
                       self.history if history is None else history,
                       self.rho_tot_0, self.ag)

    def tree_flatten(self):
        children = (self.constituents, self.ds, self.history, self.rho_tot_0, self.ag)
        return children, (self.ag is None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.constituents, obj.ds, obj.history, obj.rho_tot_0, obj.ag) = children
        return obj


# ==========================================
# Density and volume fractions
# ==========================================


def rho_tot_prev_calc(mixt):
    """rho_tot(s) = sum_j rho^j(s), from the last committed step  (Eq. 25)"""
    return sum(c.history.rho[c.history.n] for c in mixt.constituents)


@jit
def Phi_j_calc(rho_j, rho_tot):
    """phi^j = v^j/v^tot, here rho^j/rho_tot                          (Eq. 27)"""
    return rho_j / rho_tot


# ==========================================
# Nonhomeostatic degradation and deposition
#   hist holds committed cohorts 0..n; current-step values are passed in.
# ==========================================


#: One threshold for every branch of the Eq. 11/13/24 driver. Using different
#: values per function lets them switch branches at different times.
SIGMA_F_EPS = 1e-12


@jit
def sigma_f_rel(sigma_f_s, sigma_f_0, rho_tot, rho_tot_0, eps=SIGMA_F_EPS):
    """rel = [rho_tot sigma_f(s) - rho_tot(0) sigma_f(0)]
             / [rho_tot(0) sigma_f(0)]                           (Eq. 11/13/24)

    sigma_f is per unit mass, so this matches UMAT_GR_CM.for, which first
    divides by rho_coll and then weights both sides by the total density,
    and matches HCMM._sigma_f_rel.

    The denominator is omitted when sigma_f(0) = 0, which is the intended mode
    for an elastin matrix: its isochoric stress is deviatoric, so tr(sigma(G))
    vanishes. Table 2 footnote (2) covers this, and the gain then carries units
    of MPa^-1 rather than being dimensionless. The gain itself is applied by
    the caller in every branch.
    """
    ref = rho_tot_0 * sigma_f_0
    denom = jnp.where(jnp.abs(ref) < eps, 1.0, ref)
    return (rho_tot * sigma_f_s - ref) / denom


@jit
def K_exp(par, sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0, eps=SIGMA_F_EPS):
    """K_-^j(s) = (1/T^j)[1 + k_sigma_- (sigma_f(s) - sigma_f(0))/sigma_f(0)]

    Per step, using T_steps = T/ds. The denominator is dropped when
    sigma_f(0) = 0.                                                   (Eq. 13)
    """
    rel = sigma_f_rel(sigma_f_s, sigma_f_0, rho_tot, rho_tot_0, eps)
    return (1.0 / T_steps(par, ds)) * (1 + par.k_sigma_minus * rel)


@jit
def K_cumu_calc(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0):
    """K_cumu(s) = int_0^s K_-^j dtau, accumulated by the trapezoid rule:

        K_cumu(s) = K_cumu(s-1) + [K_-(s-1) + K_-(s)]/2               (Eq. 12)

    One step wide in step units, hence no ds factor.
    """
    sigma_f_0 = hist.sigma_f_0
    K_prev = K_exp(par, hist.sigma_f[hist.n], sigma_f_0, ds, rho_tot, rho_tot_0)
    K_new = K_exp(par, sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0)
    return hist.K_cumu[hist.n] + 0.5 * (K_prev + K_new)


@jit
def q_calc(hist, K_cumu_s, k):
    """q^j(s,tau) = exp(-[K_cumu(s) - K_cumu(tau)])                   (Eq. 12)

    The surviving fraction at s of material deposited at tau, for every
    retained cohort.
    """
    K_cumu_full = hist.K_cumu.at[k].set(K_cumu_s)
    return jnp.exp(-(K_cumu_s - K_cumu_full))


@jit
def rho_calc_from_sigma_f(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0,
                          eps=SIGMA_F_EPS):
    """rho^j(s) = rho^j(s-1)
                 exp[ (k_sigma_+ - k_sigma_-)/T^j
                      int_{s-1}^s (sigma_f - sigma_f(0))/sigma_f(0) dtau ]

    Closed form of the Eq. 1 mass balance with Eq. 11 and Eq. 13 substituted,
    the integral taken by the trapezoid rule over one step.
    """
    sigma_f_0 = hist.sigma_f_0
    rho_prev = hist.rho[hist.n]
    sigma_frac_prev = sigma_f_rel(hist.sigma_f[hist.n], sigma_f_0,
                                  rho_tot, rho_tot_0, eps)
    sigma_frac_new = sigma_f_rel(sigma_f_s, sigma_f_0, rho_tot, rho_tot_0, eps)
    rate = (par.k_sigma_plus - par.k_sigma_minus) / T_steps(par, ds)
    return rho_prev * jnp.exp(rate / 2.0 * (sigma_frac_prev + sigma_frac_new))


@jit
def m_j_calc(par, rho_s, sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0,
             eps=SIGMA_F_EPS):
    """m^j(s) = (rho^j(s)/T^j)[1 + k_sigma_+ (sigma_f(s) - sigma_f(0))/sigma_f(0)]

    Per step, using T_steps = T/ds. The denominator is dropped when
    sigma_f(0) = 0.                                                   (Eq. 11)
    """
    rel = sigma_f_rel(sigma_f_s, sigma_f_0, rho_tot, rho_tot_0, eps)
    return (rho_s / T_steps(par, ds)) * (1 + par.k_sigma_plus * rel)


def prestress_stress_snapshot(g_axial, elastin_kwargs, fiber_specs):
    """sigma(F=I) = rho_0^elas * material.sigma(G^elas) + sum_i rho_0^i * material.sigma(G^i)

    Snapshot stress at F=I: F_e^j = G^j exactly for every constituent, no
    cohort history/integral involved. Fiber G's use the fixed, measured g=1.1
    (Ferruzzi/Bellini); only elastin's g_axial is unknown here.
    """
    C10, K, rho0_e = elastin_kwargs['C10'], elastin_kwargs['K'], elastin_kwargs['rho_0']
    elastin_material = NeoHookean(C10=C10, K=K)
    G_e = jnp.diag(jnp.array([1.0 / jnp.sqrt(g_axial), g_axial, 1.0 / jnp.sqrt(g_axial)]))
    sigma = rho0_e * elastin_material.sigma(G_e)

    for spec in fiber_specs:
        M = jnp.asarray(spec['M'])
        M = M / jnp.linalg.norm(M)
        material = Fung(spec['k1'], spec['k2'], M)
        P = jnp.outer(M, M)
        g = spec['g']
        G_f = g * P + (1.0 / jnp.sqrt(g)) * (jnp.eye(3) - P)
        sigma = sigma + spec['rho_0'] * material.sigma(G_f)

    return sigma


def solve_prestress_g(target, elastin_kwargs, fiber_specs, axis=1, lo=1.0, hi=5.0, iters=60):
    """Solve sigma(F=I)[axis,axis] = target for elastin's axial deposition
    stretch g by bisection (axis=1 is Fig. 1's loading direction). Fiber
    deposition stretches stay fixed.
    """
    f_lo = float(prestress_stress_snapshot(lo, elastin_kwargs, fiber_specs)[axis, axis])
    f_hi = float(prestress_stress_snapshot(hi, elastin_kwargs, fiber_specs)[axis, axis])
    if (f_lo - target) * (f_hi - target) > 0:
        raise RuntimeError(f"target {target} not bracketed: f({lo})={f_lo}, f({hi})={f_hi}")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = float(prestress_stress_snapshot(mid, elastin_kwargs, fiber_specs)[axis, axis])
        if (f_mid - target) * (f_lo - target) <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return mid


# ==========================================
# Strain energy and stress
#   m_s, K_cumu_s are the current-step values; hist is committed only.
# ==========================================


@jit
def Psi_j_tot_calc(par, hist, m_s, K_cumu_s, mix_hist):
    """Psi^j(s) = int_0^s m^j(tau) q^j(s,tau) W^j(F_e^j(s,tau)) dtau  (Eq. 3)

    Trapezoid rule over the retained cohorts (Eq. 35).
    """
    k = mix_hist.n
    m_full = hist.m.at[k].set(m_s)
    q_values = q_calc(hist, K_cumu_s, k)
    F_e_history = F_e_calc(mix_hist.F, mix_hist.Fg, par.G, k)
    W_values = jax.vmap(par.material.Psi)(F_e_history)
    return jnp.trapezoid(m_full * q_values * W_values, step_axis(k, mix_hist.n_max))


@jit
def sigma_j_calc(par, hist, m_s, K_cumu_s, mix_hist, J_g):
    """Contribution of constituent j to the mixture stress:

        phi^j sigma^j = (1/J) int_0^s m^j(tau) q^j(s,tau) (dW^j/dF) F^T dtau

    Eq. 34 carries the prefactor 1/(phi^j J) and Eq. 27 multiplies by phi^j,
    so phi^j cancels and only 1/J survives. Since int m q dtau is the
    referential density rho^j (Eq. 1), this equals rho^j/J times the
    mass-averaged material stress -- the same mixture rule HCMM uses.

    Trapezoid rule over the retained cohorts (Eq. 35). With grows=False the
    growth kinematics are switched off: F_g = I.
    """
    k = mix_hist.n
    m_full = hist.m.at[k].set(m_s)
    q_values = q_calc(hist, K_cumu_s, k)

    Fg_hist = (mix_hist.Fg if par.grows
               else jnp.broadcast_to(jnp.eye(3), mix_hist.Fg.shape))

    F_e_history = F_e_calc(mix_hist.F, Fg_hist, par.G, k)
    sigma_values = jax.vmap(par.material.sigma)(F_e_history)
    weights = (m_full * q_values)[:, None, None]
    integral = jnp.trapezoid(sigma_values * weights,
                             step_axis(k, mix_hist.n_max), axis=0)
    J = jnp.linalg.det(mix_hist.F[k])
    return integral / J


# ==========================================
# Inner solve: sigma_f(s) is the unknown; rho, m, K_cumu follow from it,
#   sigma_j is the residual check.
# ==========================================


@jit
def solve_sigma_f_newton(par, hist, mix_hist, J_g, ds, rho_tot, rho_tot_0,
                         tol=1e-9, max_iter=50):
    """Close the m^j <-> sigma^j loop of sec. 2.4.1 by Newton on sigma_f.

    sigma_f is carried PER UNIT MASS -- sigma_j_calc returns rho^j/J times the
    mass-averaged stress, so dividing its trace by rho^j gives the same
    quantity HCMM scalarizes, and both models then apply the same rho_tot
    weighting in sigma_f_rel. rho_tot is lagged from the last committed step,
    as J_g is, so the per-constituent solves stay uncoupled.
    """
    sigma_f_0 = hist.sigma_f_0

    def eval_state(sigma_f_s):
        rho_s = rho_calc_from_sigma_f(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0)
        m_s = m_j_calc(par, rho_s, sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0)
        K_cumu_s = K_cumu_calc(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0)
        sigma_j = sigma_j_calc(par, hist, m_s, K_cumu_s, mix_hist, J_g)
        sigma_f_new = par.material.sigma_f(sigma_j) / rho_s
        return sigma_f_new, sigma_j, rho_s, m_s, K_cumu_s

    def residual(sigma_f_s):
        sigma_f_new, *_ = eval_state(sigma_f_s)
        return sigma_f_new - sigma_f_s

    def cond(state):
        _, r, i = state
        return (jnp.abs(r) > tol) & (i < max_iter)

    def body(state):
        sf, r, i = state
        dr = jax.grad(residual)(sf)
        dr_safe = jnp.where(jnp.abs(dr) < 1e-12, 1e-12, dr)
        sf_new = sf - r / dr_safe
        return (sf_new, residual(sf_new), i + 1)

    sf0 = hist.sigma_f[hist.n]
    sf_star, _, _ = jax.lax.while_loop(cond, body, (sf0, residual(sf0), 0))

    _, sigma_j_star, rho_star, m_star, K_cumu_star = eval_state(sf_star)
    return sf_star, sigma_j_star, rho_star, m_star, K_cumu_star


# ==========================================
# Solver entry point (shared with HCMM.py -- see jaxFEM_solver.py)
# ==========================================


@jit
def sigma_solver(mix, F):
    """sigma_tot = sum_j phi^j sigma^j at a trial F                  (Eq. 27)

    The trial cohort goes to a copy of the buffers, so mix is unchanged and an
    outer equilibrium loop may call this repeatedly. aux carries the trial
    state so commit needs no further solves.
    """
    J_g = J_g_calc(mix)
    Fg_s = mix.F_g_calc()
    mix_hist_trial = mix.history.advance(F, Fg_s)
    # Roll the constituent windows in lockstep, or cohort tau lands on
    # different slots in the two buffers.
    rolled = [c.history.roll_if_full() for c in mix.constituents]
    rho_tot = rho_tot_prev_calc(mix)

    results = [solve_sigma_f_newton(c.params, h, mix_hist_trial, J_g, mix.ds,
                                    rho_tot, mix.rho_tot_0)
               for c, h in zip(mix.constituents, rolled)]

    # phi^j already cancelled inside sigma_j_calc (Eq. 34 x Eq. 27), so the
    # constituent contributions simply add.
    sigma_total = jnp.zeros((3, 3))
    for (_, sigma_j, _, _, _) in results:
        sigma_total += sigma_j

    return sigma_total, (results, mix_hist_trial, rolled)


@jit
def commit(mix, F, aux):
    """Append the settled cohort (m^j, sigma_f^j, K_cumu^j, rho^j) to every
    constituent window and (F, F_g) to the mixture window.

    Functional: returns a new mixture, so the step stays jittable.
    """
    results, mix_hist_trial, rolled = aux
    k = mix_hist_trial.n
    constituents = [
        constituent(c.params, h.write(k, m_s, sf_s, K_cumu_s, rho_s))
        for c, h, (sf_s, _, rho_s, m_s, K_cumu_s)
        in zip(mix.constituents, rolled, results)
    ]
    return mix.replace(constituents=constituents, history=mix_hist_trial)
