"""Full constrained mixture model (FCMM), Maes & Famaey (2023) sec. 2.1.

    rho^j(s)     = rho_0^j Q^j(s) + int_0^s m^j(tau) q^j(s,tau) dtau   (Eq. 1)
    F_e^j(s,tau) = F(s) F_g(s)^-1 [F(tau) F_g(tau)^-1]^-1 G^j          (Eq. 5)
    sigma^j(s)   = 1/(phi^j J) int_0^s m^j q^j (dW^j/dF) F^T dtau      (Eq. 4)
    m^j(s)       = (rho^j/T^j)(1 + k_sigma_+ rel)                      (Eq. 11)
    q^j(s,tau)   = exp(-int_tau^s K_-^j dtau')                         (Eq. 12)

Solver interface: sigma_tot, aux = sigma_solver(state, F); state = commit(state, F, aux)
"""

import jax
import jax.numpy as jnp
from jax import jit


@jit
def F_e_calc(F, F_g, G, n):
    """F_e^j(s,tau) = F(s) F_g(s)^-1 [F(tau) F_g(tau)^-1]^-1 G^j   (Eq. 5)"""
    inner = F @ jnp.linalg.inv(F_g)
    return F[n] @ jnp.linalg.inv(F_g[n]) @ jnp.linalg.inv(inner) @ G


def F_g_iso_calc(mixt):
    """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 6)"""
    return J_g_calc(mixt) ** (1.0 / 3.0) * jnp.eye(3)


def F_g_aniso_calc(mixt, ag):
    """F_g(s) = (rho_tot(s)/rho_tot(0) - 1) ag(x)ag + I   (Eq. 7)"""
    return (J_g_calc(mixt) - 1) * jnp.outer(ag, ag) + jnp.eye(3)


def J_g_calc(mixt):
    """J_g(s) = det F_g(s) = rho_tot(s)/rho_tot(0)   (Eq. 6/7)"""
    return rho_tot_prev_calc(mixt) / mixt.rho_tot_0


def step_axis(n, n_max):
    """Trapezoid abscissa in step units, clamped at n so unfilled slots have zero width."""
    return jnp.minimum(jnp.arange(n_max), n).astype(jnp.result_type(float))


def T_steps(par, ds):
    """T_steps = T/ds"""
    return par.T / ds


def window_for(T, ds, tol=1e-3, per_cohort=False):
    """Retained cohorts n.

    tail bound:  exp(-n ds/T) / (1 - exp(-ds/T)) < tol
    per_cohort:  exp(-n ds/T) < tol   (sec. 5)
    """
    import math
    T, ds = float(T), float(ds)
    if per_cohort:
        return int(math.ceil(-math.log(tol) * T / ds))
    return int(math.ceil(-math.log(tol * (1.0 - math.exp(-ds / T))) * T / ds))


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
        """W^coll = k1/(2 k2) [exp(k2 (I4 - 1)^2) - 1],  I4 = |F M|^2   (Eq. 28)"""
        FM = F @ self.M
        I4 = jnp.dot(FM, FM)
        return (self.k1 / (2 * self.k2)) * (jnp.exp(self.k2 * (I4 - 1) ** 2) - 1)

    @jit
    def sigma(self, F):
        """sigma = (dW/dF) F^T   (Eq. 29)"""
        dW_dF = jax.grad(self.Psi)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = tr(sigma)   (Eq. 31)"""
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
        """W^elas = C10 (J_e^(-2/3) tr(F^T F) - 3) + K/2 (J_e - 1)^2   (Eq. 28)"""
        I1 = jnp.trace(F.T @ F)
        J = jnp.linalg.det(F)
        I1_inc = I1 * J ** (-2 / 3)
        return self.C10 * (I1_inc - 3) + self.K / 2 * (J - 1) ** 2

    @jit
    def sigma(self, F):
        """sigma = (dW/dF) F^T   (Eq. 29)"""
        dW_dF = jax.grad(self.Psi)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = tr(sigma)   (Eq. 31)"""
        return jnp.trace(sigma)


def _advance(buf, value, k, shift):
    """Write value at slot k, rolling out the oldest entry if shift."""
    return jnp.where(shift, jnp.roll(buf, -1, axis=0), buf).at[k].set(value)


@jax.tree_util.register_pytree_node_class
class history:
    """Rolling per-cohort buffers m^j, sigma_f^j, K_cumu^j, rho^j of one constituent."""

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
            K_cumu=z,
            rho=z.at[0].set(rho_0),
            n=jnp.asarray(0),
            sigma_f_0=jnp.asarray(sigma_f_0),
            rho_0=jnp.asarray(rho_0),
        )

    @property
    def n_max(self):
        return self.m.shape[0]

    def roll_if_full(self):
        """Shift the window if full; must stay in lockstep with mixture_history.advance."""
        shift = (self.n + 1) >= self.n_max
        roll = lambda b: jnp.where(shift, jnp.roll(b, -1, axis=0), b)
        return history(
            m=roll(self.m), sigma_f=roll(self.sigma_f),
            K_cumu=roll(self.K_cumu), rho=roll(self.rho),
            n=jnp.where(shift, self.n - 1, self.n),
            sigma_f_0=self.sigma_f_0, rho_0=self.rho_0,
        )

    def write(self, k, m_s, sigma_f_s, K_cumu_s, rho_s):
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
        """m^j(0) = rho^j(0)/T^j   (Eq. 10)"""
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
    """Rolling F(tau), F_g(tau) buffers; unfilled slots hold identity."""

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
        ag = None if ag is None else jnp.asarray(ag) / jnp.linalg.norm(jnp.asarray(ag))
        return cls(constituents, ds, mixture_history.allocate(n_max, F0),
                   sum(c.history.rho_0 for c in constituents), ag)

    def F_g_calc(self):
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


def rho_tot_prev_calc(mixt):
    """rho_tot = sum_j rho^j, last committed step   (Eq. 25)"""
    return sum(c.history.rho[c.history.n] for c in mixt.constituents)


@jit
def Phi_j_calc(rho_j, rho_tot):
    """phi^j = rho^j/rho_tot   (Eq. 27)"""
    return rho_j / rho_tot


SIGMA_F_EPS = 1e-12


@jit
def sigma_f_rel(sigma_f_s, sigma_f_0, rho_tot, rho_tot_0, eps=SIGMA_F_EPS):
    """rel = [rho_tot sigma_f(s) - rho_tot(0) sigma_f(0)] / [rho_tot(0) sigma_f(0)]   (Eq. 11/13)

    Denominator dropped when sigma_f(0) = 0 (Table 2, note 2).
    """
    ref = rho_tot_0 * sigma_f_0
    denom = jnp.where(jnp.abs(ref) < eps, 1.0, ref)
    return (rho_tot * sigma_f_s - ref) / denom


@jit
def K_exp(par, sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0, eps=SIGMA_F_EPS):
    """K_-^j(s) = (1/T^j)(1 + k_sigma_- rel)   (Eq. 13)"""
    rel = sigma_f_rel(sigma_f_s, sigma_f_0, rho_tot, rho_tot_0, eps)
    return (1.0 / T_steps(par, ds)) * (1 + par.k_sigma_minus * rel)


@jit
def K_cumu_calc(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0):
    """K_cumu(s) = K_cumu(s-1) + [K_-(s-1) + K_-(s)]/2   (Eq. 12)"""
    sigma_f_0 = hist.sigma_f_0
    K_prev = K_exp(par, hist.sigma_f[hist.n], sigma_f_0, ds, rho_tot, rho_tot_0)
    K_new = K_exp(par, sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0)
    return hist.K_cumu[hist.n] + 0.5 * (K_prev + K_new)


@jit
def q_calc(hist, K_cumu_s, k):
    """q^j(s,tau) = exp(-[K_cumu(s) - K_cumu(tau)])   (Eq. 12)"""
    K_cumu_full = hist.K_cumu.at[k].set(K_cumu_s)
    return jnp.exp(-(K_cumu_s - K_cumu_full))


@jit
def rho_calc_from_sigma_f(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0,
                          eps=SIGMA_F_EPS):
    """rho^j(s) = rho^j(s-1) exp[(k_sigma_+ - k_sigma_-)/T^j int_{s-1}^s rel dtau]   (Eq. 1 + 11 + 13)"""
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
    """m^j(s) = (rho^j(s)/T^j)(1 + k_sigma_+ rel)   (Eq. 11)"""
    rel = sigma_f_rel(sigma_f_s, sigma_f_0, rho_tot, rho_tot_0, eps)
    return (rho_s / T_steps(par, ds)) * (1 + par.k_sigma_plus * rel)


def prestress_stress_snapshot(g_axial, elastin_kwargs, fiber_specs):
    """sigma(F=I) = rho_0^elas sigma(G^elas) + sum_i rho_0^i sigma(G^i)"""
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
    """Bisection on elastin g so that sigma(F=I)[axis,axis] = target."""
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


@jit
def Psi_j_tot_calc(par, hist, m_s, K_cumu_s, mix_hist):
    """Psi^j(s) = int m^j(tau) q^j(s,tau) W^j(F_e^j(s,tau)) dtau   (Eq. 3, 35)"""
    k = mix_hist.n
    m_full = hist.m.at[k].set(m_s)
    q_values = q_calc(hist, K_cumu_s, k)
    F_e_history = F_e_calc(mix_hist.F, mix_hist.Fg, par.G, k)
    W_values = jax.vmap(par.material.Psi)(F_e_history)
    return jnp.trapezoid(m_full * q_values * W_values, step_axis(k, mix_hist.n_max))


@jit
def sigma_j_calc(par, hist, m_s, K_cumu_s, mix_hist, J_g):
    """phi^j sigma^j = (1/J) int m^j(tau) q^j(s,tau) (dW^j/dF) F^T dtau   (Eq. 27 x 34, 35)"""
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


@jit
def solve_sigma_f_newton(par, hist, mix_hist, J_g, ds, rho_tot, rho_tot_0,
                         tol=1e-9, max_iter=50):
    """Newton on sigma_f (per unit mass) closing the m^j <-> sigma^j loop (sec. 2.4.1)."""
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


@jit
def sigma_solver(mix, F):
    """sigma_tot = sum_j phi^j sigma^j at trial F, state untouched   (Eq. 27)"""
    J_g = J_g_calc(mix)
    Fg_s = mix.F_g_calc()
    mix_hist_trial = mix.history.advance(F, Fg_s)
    rolled = [c.history.roll_if_full() for c in mix.constituents]
    rho_tot = rho_tot_prev_calc(mix)

    results = [solve_sigma_f_newton(c.params, h, mix_hist_trial, J_g, mix.ds,
                                    rho_tot, mix.rho_tot_0)
               for c, h in zip(mix.constituents, rolled)]

    sigma_total = jnp.zeros((3, 3))
    for (_, sigma_j, _, _, _) in results:
        sigma_total += sigma_j

    return sigma_total, (results, mix_hist_trial, rolled)


@jit
def commit(mix, F, aux):
    """Append the settled cohort to all buffers."""
    results, mix_hist_trial, rolled = aux
    k = mix_hist_trial.n
    constituents = [
        constituent(c.params, h.write(k, m_s, sf_s, K_cumu_s, rho_s))
        for c, h, (sf_s, _, rho_s, m_s, K_cumu_s)
        in zip(mix.constituents, rolled, results)
    ]
    return mix.replace(constituents=constituents, history=mix_hist_trial)
