import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================


@jit
def F_e_calc(F, F_g, G):
    """
    F_e^j(s,tau) = F(s) F_g(s)^-1 [ F(tau) F_g(tau)^-1 ]^-1 G^j
    """
    inner = F @ jnp.linalg.inv(F_g)
    F_e = F[-1] @ jnp.linalg.inv(F_g[-1]) @ jnp.linalg.inv(inner) @ G
    return F_e


def F_g_calc(mixt):
    return (rho_tot_calc(mixt)/mixt.rho_tot_0)**(1.0/3.0) * jnp.eye(3)
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
        children = (self.k1, self.k2, self.M)
        aux_data = None
        return children, aux_data

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
        children = (self.C10, self.K)
        aux_data = None
        return children, aux_data

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


@jax.tree_util.register_pytree_node_class
class history:
    def __init__(self, m, sigma_f, K_cumu, rho):
        self.m = jnp.asarray(m)
        self.sigma_f = jnp.asarray(sigma_f)
        self.K_cumu = jnp.asarray(K_cumu)
        self.rho = jnp.asarray(rho)

    def tree_flatten(self):
        children = (self.m, self.sigma_f, self.K_cumu, self.rho)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        m, sigma_f, K_cumu, rho = children
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
        return (self.material, self.T, self.G, self.k_sigma_minus, self.k_sigma_plus, self.phi_0), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.material, obj.T, obj.G, obj.k_sigma_minus,
         obj.k_sigma_plus, obj.phi_0) = children
        return obj


class constituent:
    def __init__(self, material, rho_0, sigma_f_0, T, G, k_minus, k_plus):
        self.history = history(rho_0 / T, sigma_f_0, 0.0, rho_0)
        self.params = params(material, T, G, k_minus, k_plus)

    def update(self, sigma_f_s, m_s, rho, ds):
        sigma_0 = self.history.sigma_f[0]
        K_prev = K_exp(self.params, self.history.sigma_f[-1], sigma_0)
        K_new = K_exp(self.params, sigma_f_s, sigma_0)
        K_cumu_new = self.history.K_cumu[-1] + 0.5 * (K_prev + K_new) * ds

        self.history = history(
            m=jnp.concatenate([self.history.m, jnp.atleast_1d(m_s)]),
            sigma_f=jnp.concatenate([self.history.sigma_f, jnp.atleast_1d(sigma_f_s)]),
            K_cumu=jnp.concatenate([self.history.K_cumu, jnp.atleast_1d(K_cumu_new)]),
            rho=jnp.concatenate([self.history.rho, jnp.atleast_1d(rho)]),
        )

@jax.tree_util.register_pytree_node_class
class mixture_history:
    def __init__(self, F, Fg, s):
        self.F = jnp.asarray(F)
        self.Fg = jnp.asarray(Fg)
        self.s = jnp.asarray(s)

    def tree_flatten(self):
        children = (self.F, self.Fg, self.s)
        aux_data = None
        return children, aux_data

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

    def update(self, F_s, F_g_s, converged_states):
        self.history = mixture_history(
            F=jnp.concatenate([self.history.F, F_s[None]]),
            Fg=jnp.concatenate([self.history.Fg, F_g_s[None]]),
            s=jnp.concatenate([self.history.s, jnp.atleast_1d(self.history.s[-1] + self.ds)]),
        )
        for c, (sigma_f_s, m_s, rho_s) in zip(self.constituents, converged_states):
            c.update(sigma_f_s, m_s, rho_s, self.ds)


# ==========================================
# Material - non homeostatic
# ==========================================


@jit
def K_exp(T, k, sigma_f_s, sigma_0, eps=1e-12):
    denom = jnp.where(jnp.abs(sigma_0) < eps, 1.0, sigma_0)
    rel = jnp.where(jnp.abs(sigma_0) < eps, 0.0, k * (sigma_f_s - sigma_0) / denom)
    return (1.0 / T) * (1 + rel)


def q_j_calc(hist):
    return jnp.exp(-(hist.K_cumu[-1] - hist.K_cumu))


@jit
def m_j_calc(par, hist, eps=1e-12):
    """m(s) = rho(s)/T * (1 + k_sigma+ * (sigma_f(s)-sigma_f(0))/sigma_f(0))  (Eq. 11)"""
    denom = jnp.where(jnp.abs(hist.sigma_f[0]) < eps, 1.0, hist.sigma_f[0])
    return (hist.rho[-1] / par.T) * (1 + par.k_sigma_plus*(hist.sigma_f[-1]-hist.sigma_f[0])/denom)

def rho_calc(hist, s_hist):
    """rho(s) = integral_0^s m(tau)*q(s,tau) dtau"""
    q_values = q_j_calc(hist)
    return jnp.trapezoid(hist.m * q_values, s_hist)


def rho_tot_calc(mixt):
    return sum([c.history.rho[-1] for c in mixt.constituents])


@jit
def Phi_j_calc(rho_j, rho_tot):
    return rho_j / rho_tot

@jit
def Psi_j_tot_calc(par, hist, mix_hist, Psi_fn):
    """Psi^j(s) = integral_0^s m(tau)*q(s,tau)*W(F_s,F(tau)) dtau  (Eq. 3)"""
    q_values = q_j_calc(hist)

    F_e_history = F_e_calc(mix_hist.F, mix_hist.Fg, par.G)
    W_values = jax.vmap(Psi_fn)(F_e_history)

    integrand = hist.m * q_values * W_values
    return jnp.trapezoid(integrand, mix_hist.s)


@jit
def sigma_j_calc_from_m(par, hist, mix_hist, J_g):
    q_values = q_j_calc(hist)

    F_e_history = F_e_calc(mix_hist.F, mix_hist.Fg, par.G)
    sigma_values = jax.vmap(par.material.sigma)(F_e_history)

    weights = (hist.m * q_values)[:, None, None]
    integral = jnp.trapezoid(sigma_values * weights, mix_hist.s, axis=0)

    J = jnp.linalg.det(mix_hist.F[-1])
    kinematic_factor = (J_g / J) * (hist.rho[0] / par.phi_0)
    return kinematic_factor * integral

@jit
def rho_calc_from_sigma_f(par, hist, sigma_f):
    """
    Closed-form rho(s) for the NONHOMEOSTATIC case (Eq. 1 + Eq. 11 + Eq. 13).
    rho(s) = rho(s_prev) * exp(rate * Integral_{s_prev}^{s} sigma_frac(tau) dtau)
    """
    sigma_f_0 = hist.sigma_f[0]
    rho_prev = hist.rho[-2]

    sigma_frac_prev = (hist.sigma_f[-2] - sigma_f_0) / sigma_f_0
    sigma_frac_new  = (hist.sigma_f[-1] - sigma_f_0) / sigma_f_0

    rate = (par.k_sigma_plus - par.k_sigma_minus) / par.T
    return rho_prev * jnp.exp(rate * hist.ds / 2.0 * (sigma_frac_prev + sigma_frac_new))

@jit
def m_j_calc_from_sigma_f(par, hist, eps=1e-12):
    """m(s) = rho(s)/T * (1 + k_sigma+ * (sigma_f(s)-sigma_f(0))/sigma_f(0))  (Eq. 11)"""
    denom = jnp.where(jnp.abs(hist.sigma_f[0]) < eps, 1.0, hist.sigma_f[0])
    return (hist.rho[-1] / par.T) * (1 + par.k_sigma_plus*(hist.sigma_f[-1]-hist.sigma_f[0])/denom)



@jit
def solve_sigma_newton(par, hist, mix_hist, J_g, tol=1e-9, max_iter=50):

    def eval_m(m_trial):
        sigma_j = sigma_j_calc_from_m(par, hist, m_trial, mix_hist, J_g)
        sigma_f = par.material.sigma_f(sigma_j)
        rho_s = rho_calc_from_sigma_f(par, hist, sigma_f)
        m_new = m_j_calc_from_sigma_f(par, rho_s, sigma_f, hist.sigma_f[0])
        return m_new - m_trial

    def cond(state):
        m, r, i = state
        return (jnp.abs(r) > tol) & (i < max_iter)

    def body(state):
        m, r, i = state
        dr = jax.grad(eval_m)(m)
        dr_safe = jnp.where(jnp.abs(dr) < 1e-12, 1e-12, dr)
        m_new = m - r / dr_safe
        return (m_new, eval_m(m_new), i + 1)

    m0 = hist.m[-1]  # or hist.m[-2] if extend-after means hist has no placeholder at all yet
    state0 = (m0, eval_m(m0), 0)
    m_star, r_final, n_iter = jax.lax.while_loop(cond, body, state0)
    return m_star

def CMM_sigma_calc(F_s, mix, J_g):
    """
    Given F_s, resolve every constituent's internal state at this
    (already-extended) step and return the total mixture Cauchy stress.
    Mirrors a UMAT: takes F, returns sigma. History's last slot for
    every array is assumed already extended (tau, F_s placed in) --
    this function only OVERWRITES those slots, never appends.
    """
    mix_hist = mix.mixture_history

    mix_hist.F[-1] = F_s

    # Step 1: resolve each constituent independently
    for c in mix.constituents:
        # calculate m for current F_s
        solve_sigma_newton(c.par, c.hist, mix_hist, J_g, tol=1e-9, max_iter=50)


    rho_tot = rho_tot_calc(mixture_prop) 
    for c in  
    sigma_total += c.rho_history[-1] / rho_tot * sigma_j

    return sigma_total
