import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================


@jit
def F_e_calc(F_s, F_tau, F_g_s, F_g_tau, G):
    """
    F_e^j(s,tau) = F(s) F_g(s)^-1 [ F(tau) F_g(tau)^-1 ]^-1 G^j
    """
    inner = F_tau @ jnp.linalg.inv(F_g_tau)
    F_e = F_s @ jnp.linalg.inv(F_g_s) @ jnp.linalg.inv(inner) @ G
    return F_e


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

    @jit
    def sigma_f(sigma):
        return jnp.trace(sigma)


class constituent:
    def __init__(self, material, rho_0, sigma_f_0, T, G, k_minus, k_plus):
        self.material = material
        self.T = T
        self.G = G
        self.m_history = [rho_0 / T]
        self.rho_history = [rho_0]
        self.sigma_f_history = [sigma_f_0]
        self.K_cumu_history = [0.0]
        self.k_sigma_minus = k_minus
        self.k_sigma_plus = k_plus

    def append_converged_step(self, sigma_f_s, m_s, rho, ds):
        sigma_0 = self.sigma_f_history[0]
        K_prev = K_exp(self.T, self.k_sigma_minus, self.sigma_f_history[-1], sigma_0)
        K_new = K_exp(self.T, self.k_sigma_minus, sigma_f_s, sigma_0)
        self.K_cumu_history.append(float(self.K_cumu_history[-1] + 0.5*(K_prev+K_new)*ds))
        self.sigma_f_history.append(sigma_f_s)
        self.rho_history.append(float(rho))
        self.m_history.append(m_s)

class history:
    def __init__(self, Constituents, tau_history, F0, ds):
        self.Constituents = Constituents
        self.tau_history = tau_history
        self.F_history = [F0]
        self.tau_history = [0.0]
        self.ds = ds

    def append(self, F_s, sigma_f_s, m_s):
        self.tau_history.append(self.tau_history[-1] + self.ds)
        self.F_history.append(F_s)
        for c, (sigma_f_s, m_s) in zip(self.Constituents, sigma_f_s, m_s):
            c.append_converged_step(sigma_f_s, m_s, self.ds)


# ==========================================
# Material - homeostatic
# ==========================================

@jit
def q_homstat_calc(s, tau, T, K_cumu_history, tau_history):
    """
    q^j(s, tau) = exp( -(s - tau) / T )   (Eq. 8)
    """
    return jnp.exp(-(s - tau) / T)


# ==========================================
# Material - non homeostatic
# ==========================================


@jit
def K_exp(T, k, sigma_f_s, sigma_0, eps=1e-12):
    denom = jnp.where(jnp.abs(sigma_0) < eps, 1.0, sigma_0)
    rel = jnp.where(jnp.abs(sigma_0) < eps, 0.0, k * (sigma_f_s - sigma_0) / denom)
    return (1.0 / T) * (1 + rel)


def q_calc(s, tau, K_cumu_history, tau_history):
    cumu = jnp.asarray(K_cumu_history)
    cumu_s = jnp.interp(s, tau_history, cumu)
    cumu_tau = jnp.interp(jnp.asarray(tau), tau_history, cumu)
    return jnp.exp(-(cumu_s - cumu_tau))


def m_calc(constituent, rho_s, sigma_f_s, sigma_f_0, eps=1e-12):
    """
    m^j(s) = rho^j(s)/T^j * (1 + k_sigma+ * (sigma_f(s)-sigma_f(0))/sigma_f(0))
    (Eq. 11)
    """
    if abs(sigma_f_0) < eps:
        return (rho_s / constituent.T) * (1 + constituent.k_sigma_plus * sigma_f_s)
    else:
        return (rho_s / constituent.T) * (
            1 + constituent.k_sigma_plus * (sigma_f_s - sigma_f_0) / sigma_f_0
        )


def rho_calc(s, constituent, tau_history):
    return jnp.interp(s, tau_history, jnp.asarray(constituent.rho_history))



def phi_calc(constituent, all_constituents, s, tau_history, q):
    """phi^j(s) = rho^j(s) / rho^tot(s), assuming equal intrinsic density across constituents."""
    rho_j = rho_calc(s, constituent, tau_history, q)
    rho_tot = sum(rho_calc(s, c, tau_history, q) for c in all_constituents)
    return rho_j / rho_tot


def Psi_j_tot_calc(constituent, s, F_s, F_history, tau_history, q):
    """
    Psi^j(s) = integral_0^s m^j(tau) * q^j(s,tau) * W(F_s, F(tau)) dtau
    (Eq. 3) assuming Q(s) = q(s,0)
    """
    tau_history = jnp.asarray(tau_history)
    m_history = jnp.asarray(constituent.m_history)

    q_values = q(s, tau_history, constituent.T, constituent.K_cumu_history, tau_history)

    F_e_history = jax.vmap(F_e_calc, in_axes=(None, 0, None, None, None))(
        F_s, F_history, jnp.eye(3), jnp.eye(3), constituent.G
    )
    W_values = jax.vmap(constituent.material.Psi)(F_e_history)

    integrand = m_history * q_values * W_values
    return jnp.trapezoid(integrand, tau_history)


def sigma_j_tot_calc(constituent, s, F_s, F_history, tau_history, q):
    tau_history = jnp.asarray(tau_history)
    m_history = jnp.asarray(constituent.m_history)

    q_values = q(s, tau_history, constituent.T, constituent.K_cumu_history, tau_history)
    kirchhoff_values = jax.vmap(constituent.material.sigma, in_axes=(0))(
        constituent.material.sigma, constituent.G, F_s, F_history
    )
    integrand = m_history * q_values * kirchhoff_values
    return 1/(jnp.det(F_s)*phi_s)*jnp.trapezoid(integrand, tau_history)
