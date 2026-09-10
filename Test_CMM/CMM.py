import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================

@jit
def C_calc(F):
    """C = F^T . F"""
    return F.T @ F

@jit
def I4_calc(C, e=(1.0, 0.0, 0.0)):
    """I4 = e . C . e"""
    e = jnp.array(e)
    e = e / jnp.linalg.norm(e)
    return e @ C @ e

@jit
def I1_calc(C, e=(1.0, 0.0, 0.0)):
    """I1 = tr(C)"""
    return jnp.trace(C)




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


def fung_strain_energy(k1, k2, M):
    """W(I4) = k1/(2*k2) * (exp(k2*(I4-1)**2) - 1)"""
    @jit
    def _calc(F):
        C = C_calc(F)
        I4 = I4_calc(C, M)
        return (k1 / (2 * k2)) * (jnp.exp(k2 * (I4 - 1) ** 2) - 1)
    return _calc

def fung_stress(k1, k2, M):
    @jit
    def _calc(F_e, J, J_g, rho_e, rho_t, Phi_e):
        F_eM = F_e @ M
        I4 = jnp.dot(F_eM, F_eM)
        W_I4 = k1 * (I4 - 1) * jnp.exp(k2 * (I4 - 1)**2)
        dyad_prod = jnp.outer(F_eM, F_eM)
        return 2 * rho_e / (Phi_e * J) * W_I4 * dyad_prod
    return _calc


def neoHookean_strain_energy(C10, K):
    """W(I4) = C10(I1_bar - 3) +K/2*(J-1)^2"""
    @jit
    def _calc(F):
        C = C_calc(F)
        I1 = I1_calc(C)
        J = jnp.linalg.det(F)
        I1_inc = I1*J**(-2/3)
        return C10*(I1_inc-3)+K/2*(J-1)**2
    return _calc

def neoHookean_stress(C10, K):
    """W(I4) = C10(I1_bar - 3) +K/2*(J-1)^2"""
    @jit
    def _calc(F_e, J, J_g, rho_e, rho_t, Phi_e):
        J_e = jnp.linalg.det(F_e)
        F_e_bar = F_e * J_e**(-1/3)
        B_e_bar = F_e_bar @ F_e_bar.T
        I1_e_bar = jnp.trace(B_e_bar)
        inc_term = 2*C10*rho_e/(Phi_e*J)*(B_e_bar-1/3*I1_e_bar*jnp.eye(3))
        comp_term = K*rho_t/J_g*(J_e-1)*jnp.eye(3)
        return inc_term + comp_term
    return _calc

class constituent:
    def __init__(self, rho_0, sigma_f_0, T, G, W, k_minus, k_plus):
        self.T = T
        self.G = G
        self.m_history = [rho_0/T]
        self.rho_history = [rho_0]
        self.sigma_f_history = [sigma_f_0]
        self.K_cumu_history = [0.0] 
        self.k_sigma_minus = k_minus
        self.k_sigma_plus = k_plus
        self.W = W


    def append_converged_step(self, sigma_f_s, ds):
        sigma_0 = self.sigma_f_history[0]
        K_prev = K_exp(self.T, self.k_sigma_minus, self.sigma_f_history[-1], sigma_0)
        K_new = K_exp(self.T, self.k_sigma_minus, sigma_f_s, sigma_0)
        increment = 0.5 * (K_prev + K_new) * ds
        self.K_cumu_history.append(float(self.K_cumu_history[-1] + increment))
        self.sigma_f_history.append(sigma_f_s)


class history:
    def __init__(self, Constituents, tau_history):
        self.Constituents = Constituents
        self.tau_history = tau_history



def rho_calc(s, constituent, tau_history, Q, q):
    """
    rho^j(s) = rho_0^j * Q^j(s) + integral_0^s m^j(tau) q^j(s,tau) dtau
    (Eq. 1)
    """
    tau_history = jnp.asarray(tau_history)
    m_history = jnp.asarray(constituent.m_history)
 
    term1 = constituent.rho_history[0] * Q(s, constituent.T, constituent.K_cumu_history, tau_history)

    integrand = m_history * q(s, tau_history, constituent.T, constituent.K_cumu_history, tau_history)
    term2 = jnp.trapezoid(integrand, tau_history)
 
    return term1 + term2

# ==========================================
# Material - homeostatic
# ==========================================


@jit
def Q_homstat_calc(s, T, K_cumu_history, tau_history):
    """
    Q^j(s) = exp(-s/T)   (Eq. 8)
    """
    return jnp.exp(-s / T)


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


def Q_calc(s, T, K_cumu_history, tau_history):
    cumu = jnp.asarray(K_cumu_history)
    return jnp.exp(-jnp.interp(s, tau_history, cumu))

    


def q_calc(s, tau, T, K_cumu_history, tau_history):
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
 



def W_calc(W, G, F_s, F_tau):
    """W(F_e)."""
    F_e = F_e_calc(F_s, F_tau, jnp.eye(3), jnp.eye(3), G)
    return W(F_e)

 
def Psi_calc(constituent, s, F_s, F_history, tau_history, Q, q):
    """
    Psi^j(s) = rho_0^j * Q^j(s) * W(F_s, F(0))
               + integral_0^s m^j(tau) * q^j(s,tau) * W(F_s, F(tau)) dtau
    (Eq. 3)
    """
    tau_history = jnp.asarray(tau_history)
    m_history = jnp.asarray(constituent.m_history)
 
    first_term = constituent.rho_history[0] * Q(s, constituent.T, constituent.K_cumu_history, tau_history) * \
        W_calc(constituent.W, constituent.G, F_s, F_history[0])
 
    q_values = q(s, tau_history, constituent.T, constituent.K_cumu_history, tau_history)
    W_values = jax.vmap(W_calc, in_axes=(None, None, None, 0))(
        constituent.W, constituent.G , F_s, F_history
    )
    integrand = m_history * q_values * W_values
    second_term = jnp.trapezoid(integrand, tau_history)
 
    return first_term + second_term






