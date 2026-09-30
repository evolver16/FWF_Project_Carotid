"""FCMM core: cohort histories, constituents, mixture, stress and the commit of one G&R step.

    rho^j(s)     = rho_0^j Q^j(s) + int_0^s m^j(tau) q^j(s,tau) dtau   (Eq. 1)
    F_e^j(s,tau) = F(s) F_g(s)^-1 [F(tau) F_g(tau)^-1]^-1 G^j          (Eq. 5)
    sigma^j(s)   = 1/(phi^j J) int_0^s m^j q^j (dW^j/dF) F^T dtau      (Eq. 4)
    q^j(s,tau)   = exp(-int_tau^s K_-^j dtau')                         (Eq. 12)

Solver interface: sigma_tot, aux = sigma_solver(state, F); state = commit(state, F, aux)
"""

import jax
import jax.numpy as jnp
from jax import jit
from fem.tensor3 import det3, inv3


def pytree(data, static=()):
    """Register a class as pytree: data attributes are leaves, static ones aux data; adds replace(**fields)"""
    def wrap(cls):
        def flatten(obj):
            return tuple(getattr(obj, k) for k in data), tuple(getattr(obj, k) for k in static)

        def unflatten(aux, children):
            obj = cls.__new__(cls)
            for k, v in zip(data, children):
                setattr(obj, k, v)
            for k, v in zip(static, aux):
                setattr(obj, k, v)
            return obj

        def replace(self, **fields):
            children, aux = flatten(self)
            obj = unflatten(aux, children)
            for k, v in fields.items():
                setattr(obj, k, v)
            return obj

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        cls.replace = replace
        return cls
    return wrap


@jit
def F_e_calc(F, F_g, R, G, n):
    """F_e^j(s,tau) = F(s) F_g(s)^-1 [F(tau) F_g(tau)^-1]^-1 R(tau) G^j   (Eq. 5, deposited in the rotated frame)"""
    inner = F @ inv3(F_g)
    return F[n] @ inv3(F_g[n]) @ inv3(inner) @ R @ G


@jit
def polar_rotation(F, n_iter=12):
    """F = R U,  R_{k+1} = (R_k + R_k^-T)/2 from R_0 = F, smooth at repeated singular values"""
    return jax.lax.fori_loop(0, n_iter, lambda _, R: 0.5 * (R + inv3(R).T), F)


def step_axis(n, n_max):
    """Trapezoid abscissa in step units, clamped at n so unfilled slots have zero width."""
    return jnp.minimum(jnp.arange(n_max), n).astype(jnp.result_type(float))


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


def _advance(buf, value, k, shift):
    """Write value at slot k, rolling out the oldest entry if shift."""
    return jnp.where(shift, jnp.roll(buf, -1, axis=0), buf).at[k].set(value)


@pytree(("m", "sigma_f", "K_cumu", "rho", "n", "sigma_f_0", "rho_0"))
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


@pytree(("material", "G", "production", "removal"), static=("grows",))
class params:
    def __init__(self, material, G, production=None, removal=None, grows=True):
        """production/removal: laws from fcmm.production / fcmm.removal (None: no turnover, e.g. elastin)"""
        self.material = material
        self.G = jnp.asarray(G)
        self.production = production
        self.removal = removal
        self.grows = grows

    @property
    def remodels(self):
        """static: None-ness of the laws is part of the pytree structure"""
        return self.production is not None or self.removal is not None


@pytree(("params", "history"))
class constituent:
    def __init__(self, params, history):
        self.params = params
        self.history = history

    @classmethod
    def allocate(cls, params, rho_0, sigma_f_0, n_max, ds):
        """m^j(0) = production at rel = 0, e.g. rho^j(0)/T^j   (Eq. 10)"""
        m_0 = params.production.m(rho_0, sigma_f_0, sigma_f_0, ds, 1.0, 1.0) if params.remodels else 0.0
        return cls(params, history.allocate(n_max, m_0, sigma_f_0, rho_0))


@pytree(("F", "Fg", "R", "n"))
class mixture_history:
    """Rolling F(tau), F_g(tau), R(tau) buffers; unfilled slots hold identity."""

    def __init__(self, F, Fg, R, n):
        self.F = F
        self.Fg = Fg
        self.R = R
        self.n = n

    @classmethod
    def allocate(cls, n_max, F0):
        eye = jnp.broadcast_to(jnp.eye(3), (n_max, 3, 3))
        F0 = jnp.asarray(F0)
        return cls(F=eye.at[0].set(F0), Fg=eye, R=eye.at[0].set(polar_rotation(F0)),
                   n=jnp.asarray(0))

    @property
    def n_max(self):
        return self.F.shape[0]

    def advance(self, F_s, Fg_s):
        shift = (self.n + 1) >= self.n_max
        k = jnp.minimum(self.n + 1, self.n_max - 1)
        return mixture_history(F=_advance(self.F, F_s, k, shift),
                               Fg=_advance(self.Fg, Fg_s, k, shift),
                               R=_advance(self.R, polar_rotation(F_s), k, shift), n=k)


@pytree(("constituents", "ds", "history", "rho_tot_0", "growth"))
class mixture:
    def __init__(self, constituents, ds, history, rho_tot_0, growth):
        """growth: law from fcmm.growth, F_g from J_g = rho_tot / rho_tot_0"""
        self.constituents = constituents
        self.ds = ds
        self.history = history
        self.rho_tot_0 = rho_tot_0
        self.growth = growth

    @classmethod
    def allocate(cls, constituents, F0, ds, n_max, growth):
        return cls(constituents, ds, mixture_history.allocate(n_max, F0),
                   sum(c.history.rho_0 for c in constituents), growth)

    def F_g_calc(self):
        return self.growth.F_g(J_g_calc(self))


def rho_tot_prev_calc(mixt):
    """rho_tot = sum_j rho^j, last committed step   (Eq. 25)"""
    return sum(c.history.rho[c.history.n] for c in mixt.constituents)


def J_g_calc(mixt):
    """J_g(s) = det F_g(s) = rho_tot(s)/rho_tot(0)   (Eq. 6/7)"""
    return rho_tot_prev_calc(mixt) / mixt.rho_tot_0


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
def K_cumu_calc(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0):
    """K_cumu(s) = K_cumu(s-1) + [K_-(s-1) + K_-(s)]/2   (Eq. 12)"""
    sigma_f_0 = hist.sigma_f_0
    K_prev = par.removal.K(hist.sigma_f[hist.n], sigma_f_0, ds, rho_tot, rho_tot_0)
    K_new = par.removal.K(sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0)
    return hist.K_cumu[hist.n] + 0.5 * (K_prev + K_new)


@jit
def q_calc(hist, K_cumu_s, k):
    """q^j(s,tau) = exp(-[K_cumu(s) - K_cumu(tau)])   (Eq. 12)"""
    K_cumu_full = hist.K_cumu.at[k].set(K_cumu_s)
    return jnp.exp(-(K_cumu_s - K_cumu_full))


@jit
def rho_calc_from_sigma_f(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0):
    """rho^j(s) = rho^j(s-1) exp[int_{s-1}^s (m^j/rho^j - K_-^j) dtau]   (Eq. 1, trapezoid)"""
    sigma_f_0 = hist.sigma_f_0
    net = lambda sf: (par.production.m(1.0, sf, sigma_f_0, ds, rho_tot, rho_tot_0)
                      - par.removal.K(sf, sigma_f_0, ds, rho_tot, rho_tot_0))
    return hist.rho[hist.n] * jnp.exp(0.5 * (net(hist.sigma_f[hist.n]) + net(sigma_f_s)))


@jit
def cohort_sigmas(par, mix_hist):
    """sigma(F_e^j(s,tau)) for every retained cohort   (Eq. 5, 29)"""
    Fg_hist = (mix_hist.Fg if par.grows
               else jnp.broadcast_to(jnp.eye(3), mix_hist.Fg.shape))
    F_e_history = F_e_calc(mix_hist.F, Fg_hist, mix_hist.R, par.G, mix_hist.n)
    return jax.vmap(par.material.sigma)(F_e_history)


def cohort_weights(hist, m_s, K_cumu_s, k):
    """m^j(tau) q^j(s,tau)   (Eq. 35)"""
    return hist.m.at[k].set(m_s) * q_calc(hist, K_cumu_s, k)


@jit
def sigma_j_calc(par, hist, m_s, K_cumu_s, mix_hist, J_g, sigma_values=None):
    """phi^j sigma^j = (1/J) int m^j(tau) q^j(s,tau) (dW^j/dF) F^T dtau   (Eq. 27 x 34, 35)"""
    k = mix_hist.n
    if sigma_values is None:
        sigma_values = cohort_sigmas(par, mix_hist)
    weights = cohort_weights(hist, m_s, K_cumu_s, k)[:, None, None]
    integral = jnp.trapezoid(sigma_values * weights, step_axis(k, mix_hist.n_max), axis=0)
    return integral / det3(mix_hist.F[k])


@jit
def sigma_j_elastic(par, hist, mix_hist):
    """phi^j sigma^j = (rho^j/J) sigma(F(s) F_g(s)^-1 G^j),  Q^j = 1   (Eq. 4 first term, 5)"""
    k = mix_hist.n
    F, J = mix_hist.F[k], det3(mix_hist.F[k])
    F_g = mix_hist.Fg[k] if par.grows else jnp.eye(3)
    sigma_mat = par.material.sigma(F @ inv3(F_g) @ par.G)
    rho = hist.rho[hist.n]
    return (par.material.sigma_f(sigma_mat) / J, rho / J * sigma_mat, rho,
            jnp.zeros(()), hist.K_cumu[hist.n])


@jit
def solve_sigma_f_newton(par, hist, mix_hist, J_g, ds, rho_tot, rho_tot_0,
                         tol=1e-9, max_iter=50):
    """Newton on sigma_f (per unit mass) closing the m^j <-> sigma^j loop (sec. 2.4.1); implicit gradient via custom_root."""
    sigma_f_0 = hist.sigma_f_0
    k = mix_hist.n
    sigma_values = cohort_sigmas(par, mix_hist)
    tr_values = jax.vmap(par.material.sigma_f)(sigma_values)
    axis = step_axis(k, mix_hist.n_max)
    J = det3(mix_hist.F[k])

    def eval_state(sigma_f_s):
        rho_s = rho_calc_from_sigma_f(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0)
        m_s = par.production.m(rho_s, sigma_f_s, sigma_f_0, ds, rho_tot, rho_tot_0)
        K_cumu_s = K_cumu_calc(par, hist, sigma_f_s, ds, rho_tot, rho_tot_0)
        w = cohort_weights(hist, m_s, K_cumu_s, k)
        sigma_f_new = jnp.trapezoid(w * tr_values, axis) / (J * rho_s)
        return sigma_f_new, rho_s, m_s, K_cumu_s

    def residual(sigma_f_s):
        sigma_f_new, *_ = eval_state(sigma_f_s)
        return sigma_f_new - sigma_f_s

    def newton(f, sf0):
        def body(state):
            sf, r, i = state
            dr = jax.grad(f)(sf)
            sf = sf - r / jnp.where(jnp.abs(dr) < 1e-12, 1e-12, dr)
            return sf, f(sf), i + 1

        return jax.lax.while_loop(lambda st: (jnp.abs(st[1]) > tol) & (st[2] < max_iter), body,
                                  (sf0, f(sf0), 0))[0]

    sf_star = jax.lax.custom_root(residual, hist.sigma_f[hist.n], newton,
                                  lambda g, y: y / g(jnp.ones_like(y)))

    _, rho_star, m_star, K_cumu_star = eval_state(sf_star)
    sigma_j_star = sigma_j_calc(par, hist, m_star, K_cumu_star, mix_hist, J_g, sigma_values)
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
               if c.params.remodels else sigma_j_elastic(c.params, h, mix_hist_trial)
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
