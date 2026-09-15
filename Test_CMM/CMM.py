import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================


@jit
def F_e_calc(F, F_g, G):
    """F_e^j(s,tau) = F(s) F_g(s)^-1 [ F(tau) F_g(tau)^-1 ]^-1 G^j"""
    inner = F @ jnp.linalg.inv(F_g)
    return F[-1] @ jnp.linalg.inv(F_g[-1]) @ jnp.linalg.inv(inner) @ G


def F_g_calc(mixt):
    """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 6, isotropic)"""
    return J_g_calc(mixt) ** (1.0 / 3.0) * jnp.eye(3)


def J_g_calc(mixt):
    """J_g(s) = det F_g(s) = rho_tot(s)/rho_tot(0), lagged from last committed step"""
    return rho_tot_prev_calc(mixt) / mixt.rho_tot_0


# ==========================================
# Material
# ==========================================


@jax.tree_util.register_pytree_node_class
class Fung:
    def __init__(self, k1, k2, M):
        self.k1 = k1
        self.k2 = k2
        M = jnp.asarray(M, dtype=jnp.float32)
        self.M = M / jnp.linalg.norm(M)

    def tree_flatten(self):
        return (self.k1, self.k2, self.M), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def Psi(self, F):
        """W(I4) = k1/(2*k2) * (exp(k2*(I4-1)**2) - 1)"""
        FM = F @ self.M
        I4 = jnp.dot(FM, FM)
        return (self.k1 / (2 * self.k2)) * (jnp.exp(self.k2 * (I4 - 1) ** 2) - 1)

    @jit
    def sigma(self, F):
        dW_dF = jax.grad(self.Psi)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
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
        I1 = jnp.trace(F.T @ F)
        J = jnp.linalg.det(F)
        I1_inc = I1 * J ** (-2 / 3)
        return self.C10 * (I1_inc - 3) + self.K / 2 * (J - 1) ** 2

    @jit
    def sigma(self, F):
        dW_dF = jax.grad(self.Psi)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        return jnp.trace(sigma)


# ==========================================
# State containers
# ==========================================


@jax.tree_util.register_pytree_node_class
class history:
    def __init__(self, m, sigma_f, K_cumu, rho):
        self.m = jnp.atleast_1d(jnp.asarray(m))
        self.sigma_f = jnp.atleast_1d(jnp.asarray(sigma_f))
        self.K_cumu = jnp.atleast_1d(jnp.asarray(K_cumu))
        self.rho = jnp.atleast_1d(jnp.asarray(rho))

    def tree_flatten(self):
        return (self.m, self.sigma_f, self.K_cumu, self.rho), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.m, obj.sigma_f, obj.K_cumu, obj.rho = children
        return obj


@jax.tree_util.register_pytree_node_class
class params:
    def __init__(self, material, T, G, k_minus, k_plus, phi_0):
        self.material = material
        self.T = jnp.asarray(T)
        self.G = jnp.asarray(G)
        self.k_sigma_minus = jnp.asarray(k_minus)
        self.k_sigma_plus = jnp.asarray(k_plus)
        self.phi_0 = jnp.asarray(phi_0)

    def tree_flatten(self):
        return (self.material, self.T, self.G, self.k_sigma_minus,
                self.k_sigma_plus, self.phi_0), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.material, obj.T, obj.G, obj.k_sigma_minus,
         obj.k_sigma_plus, obj.phi_0) = children
        return obj


class constituent:
    def __init__(self, params, rho_0, sigma_f_0):
        self.params = params
        self.history = history(rho_0 / params.T, sigma_f_0, 0.0, rho_0)

    def commit(self, sigma_f_s, m_s, rho_s, K_cumu_s):
        self.history = history(
            m=jnp.concatenate([self.history.m, jnp.atleast_1d(m_s)]),
            sigma_f=jnp.concatenate([self.history.sigma_f, jnp.atleast_1d(sigma_f_s)]),
            K_cumu=jnp.concatenate([self.history.K_cumu, jnp.atleast_1d(K_cumu_s)]),
            rho=jnp.concatenate([self.history.rho, jnp.atleast_1d(rho_s)]),
        )


@jax.tree_util.register_pytree_node_class
class mixture_history:
    def __init__(self, F, Fg, s):
        self.F = jnp.asarray(F)
        self.Fg = jnp.asarray(Fg)
        self.s = jnp.asarray(s)

    def tree_flatten(self):
        return (self.F, self.Fg, self.s), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.F, obj.Fg, obj.s = children
        return obj


class mixture:
    def __init__(self, constituents, F0, ds):
        self.constituents = constituents
        self.history = mixture_history(
            F=jnp.asarray(F0)[None],
            Fg=jnp.eye(3)[None],
            s=jnp.array([0.0]),
        )
        self.ds = ds
        self.rho_tot_0 = sum(c.history.rho[0] for c in constituents)

    def commit(self, F_s, F_g_s):
        self.history = mixture_history(
            F=jnp.concatenate([self.history.F, F_s[None]]),
            Fg=jnp.concatenate([self.history.Fg, F_g_s[None]]),
            s=jnp.concatenate([self.history.s, jnp.atleast_1d(self.history.s[-1] + self.ds)]),
        )


# ==========================================
# Density and volume fractions
# ==========================================


def rho_tot_prev_calc(mixt):
    return sum(c.history.rho[-1] for c in mixt.constituents)


@jit
def Phi_j_calc(rho_j, rho_tot):
    return rho_j / rho_tot


# ==========================================
# Nonhomeostatic degradation and deposition
#   hist holds committed cohorts (0 .. s-1); current-step values are passed in.
# ==========================================


@jit
def K_exp(par, sigma_f_s, sigma_f_0, eps=1e-12):
    """K_-(s) = 1/T * (1 + k_sigma- * (sigma_f(s)-sigma_f(0))/sigma_f(0))   (Eq. 13)"""
    denom = jnp.where(jnp.abs(sigma_f_0) < eps, 1.0, sigma_f_0)
    rel = jnp.where(jnp.abs(sigma_f_0) < eps, 0.0,
                    par.k_sigma_minus * (sigma_f_s - sigma_f_0) / denom)
    return (1.0 / par.T) * (1 + rel)


@jit
def K_cumu_calc(par, hist, sigma_f_s, ds):
    """K_cumu(s) = K_cumu(s-1) + trapezoid step of K_-   (cumulative of Eq. 12)"""
    sigma_f_0 = hist.sigma_f[0]
    K_prev = K_exp(par, hist.sigma_f[-1], sigma_f_0)
    K_new = K_exp(par, sigma_f_s, sigma_f_0)
    return hist.K_cumu[-1] + 0.5 * (K_prev + K_new) * ds


@jit
def q_calc(hist, K_cumu_s):
    """q(s,tau) = exp(-(K_cumu(s) - K_cumu(tau)))   for every committed cohort tau (Eq. 12)"""
    K_cumu_full = jnp.concatenate([hist.K_cumu, jnp.atleast_1d(K_cumu_s)])
    return jnp.exp(-(K_cumu_full[-1] - K_cumu_full))


@jit
def rho_calc_from_sigma_f(par, hist, sigma_f_s, ds):
    """rho(s) = rho(s-1) * exp((k_sigma+ - k_sigma-)/T * integral_{s-1}^{s} sigma_frac dtau)

    Closed form of Eq. 1 with Eq. 11 and Eq. 13 substituted into the mass balance.
    """
    sigma_f_0 = hist.sigma_f[0]
    rho_prev = hist.rho[-1]
    sigma_frac_prev = (hist.sigma_f[-1] - sigma_f_0) / sigma_f_0
    sigma_frac_new = (sigma_f_s - sigma_f_0) / sigma_f_0
    rate = (par.k_sigma_plus - par.k_sigma_minus) / par.T
    return rho_prev * jnp.exp(rate * ds / 2.0 * (sigma_frac_prev + sigma_frac_new))


@jit
def m_j_calc(par, rho_s, sigma_f_s, sigma_f_0, eps=1e-12):
    """m(s) = rho(s)/T * (1 + k_sigma+ * (sigma_f(s)-sigma_f(0))/sigma_f(0))   (Eq. 11)"""
    denom = jnp.where(jnp.abs(sigma_f_0) < eps, 1.0, sigma_f_0)
    rel = jnp.where(jnp.abs(sigma_f_0) < eps, sigma_f_s,
                    par.k_sigma_plus * (sigma_f_s - sigma_f_0) / denom)
    return (rho_s / par.T) * (1 + rel)


# ==========================================
# Strain energy and stress
#   m_s, K_cumu_s are the current-step values; hist is committed only.
# ==========================================


@jit
def Psi_j_tot_calc(par, hist, m_s, K_cumu_s, mix_hist):
    """Psi^j(s) = integral_0^s m(tau)*q(s,tau)*W(F_e^j(s,tau)) dtau   (Eq. 3)"""
    m_full = jnp.concatenate([hist.m, jnp.atleast_1d(m_s)])
    q_values = q_calc(hist, K_cumu_s)
    F_e_history = F_e_calc(mix_hist.F, mix_hist.Fg, par.G)
    W_values = jax.vmap(par.material.Psi)(F_e_history)
    return jnp.trapezoid(m_full * q_values * W_values, mix_hist.s)


@jit
def sigma_j_calc(par, hist, m_s, K_cumu_s, mix_hist, J_g):
    """sigma^j(s) = (J_g/J)(rho_0^j/phi_0^j) * integral_0^s m*q*sigma(F_e^j) dtau   (Eq. 4/29/30)"""
    m_full = jnp.concatenate([hist.m, jnp.atleast_1d(m_s)])
    q_values = q_calc(hist, K_cumu_s)
    F_e_history = F_e_calc(mix_hist.F, mix_hist.Fg, par.G)
    sigma_values = jax.vmap(par.material.sigma)(F_e_history)
    weights = (m_full * q_values)[:, None, None]
    integral = jnp.trapezoid(sigma_values * weights, mix_hist.s, axis=0)
    J = jnp.linalg.det(mix_hist.F[-1])
    kinematic_factor = (J_g / J) * (hist.rho[0] / par.phi_0)
    return kinematic_factor * integral


# ==========================================
# Inner solve: sigma_f(s) is the unknown; rho, m, K_cumu follow from it,
#   sigma_j is the residual check.
# ==========================================


@jit
def solve_sigma_f_newton(par, hist, mix_hist, J_g, tol=1e-9, max_iter=50):
    ds = mix_hist.s[-1] - mix_hist.s[-2]
    sigma_f_0 = hist.sigma_f[0]

    def eval_state(sigma_f_s):
        rho_s = rho_calc_from_sigma_f(par, hist, sigma_f_s, ds)
        m_s = m_j_calc(par, rho_s, sigma_f_s, sigma_f_0)
        K_cumu_s = K_cumu_calc(par, hist, sigma_f_s, ds)
        sigma_j = sigma_j_calc(par, hist, m_s, K_cumu_s, mix_hist, J_g)
        sigma_f_new = par.material.sigma_f(sigma_j)
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

    sf0 = hist.sigma_f[-1]
    sf_star, _, _ = jax.lax.while_loop(cond, body, (sf0, residual(sf0), 0))

    _, sigma_j_star, rho_star, m_star, K_cumu_star = eval_state(sf_star)
    return sf_star, sigma_j_star, rho_star, m_star, K_cumu_star


# ==========================================
# Mixture stress (UMAT entry point)
#   mix / constituent histories are already extended by one step (F_s, tau placed).
# ==========================================


def CMM_sigma_calc(F_s, mix):
    mix.history.F = mix.history.F.at[-1].set(F_s)
    J_g = J_g_calc(mix)

    sigma_j_list = []
    for c in mix.constituents:
        sf_s, sigma_j, rho_s, m_s, K_cumu_s = solve_sigma_f_newton(
            c.params, c.history, mix.history, J_g)
        c.commit(sf_s, m_s, rho_s, K_cumu_s)
        sigma_j_list.append(sigma_j)

    rho_tot = sum(c.history.rho[-1] for c in mix.constituents)

    sigma_total = jnp.zeros((3, 3))
    for c, sigma_j in zip(mix.constituents, sigma_j_list):
        sigma_total += Phi_j_calc(c.history.rho[-1], rho_tot) * sigma_j

    return sigma_total