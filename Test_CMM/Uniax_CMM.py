import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================

@jit
def F_calc_uniax(lam, e=(1.0, 0.0, 0.0)):
    """F = lam*(e⊗e) + lam**(-1/2)*(I - e⊗e)"""
    e = jnp.array(e)
    e = e / jnp.linalg.norm(e)
    I = jnp.eye(3)
    e_dyad_e = jnp.outer(e, e)
    return lam * e_dyad_e + lam ** (-0.5) * (I - e_dyad_e)


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

class constituent:
    def __init__(self, rho_0, sigma_0, T, G, W, k_minus, k_plus):
        self.T = T
        self.G = G
        self.m_history = [rho_0/T]
        self.rho_history = [rho_0]
        self.sigma_history = [sigma_0]
        self.W = W

class history:
    def __init__(self, Constituents, tau_history):
        self.Constituents = Constituents
        self.tau_history = tau_history


def fung_strain_energy(c1, c2, e):
    """W(I4) = c1/(2*c2) * (exp(c2*(I4-1)**2) - 1)"""
    @jit
    def _calc(F):
        C = C_calc(F)
        I4 = I4_calc(C, e)
        return (c1 / (2 * c2)) * (jnp.exp(c2 * (I4 - 1) ** 2) - 1)
    return _calc


def rho_calc(s, constituent, tau_history, Q, q):
    """
    rho^j(s) = rho_0^j * Q^j(s) + integral_0^s m^j(tau) q^j(s,tau) dtau
    (Eq. 1)
    """
    tau_history = jnp.asarray(tau_history)
    m_history = jnp.asarray(constituent.m_history)
 
    term1 = constituent.rho_history[0] * Q(s, constituent.T)

    integrand = m_history * q(s, tau_history, constituent.T)
    term2 = jnp.trapezoid(integrand, tau_history)
 
    return term1 + term2

# ==========================================
# Material - homeostatic
# ==========================================



@jit
def Q_homstat_calc(s,T):
    """
    Q^j(s) = exp(-s/T)
    (Eq. 8)
    """
    return jnp.exp(-s / T)


@jit
def q_homstat_calc(s,tau,T):
    """
    q^j(s, tau) = exp( -(s - tau) / T )
    (Eq. 8)
    """
    return jnp.exp(-(s - tau) / T)



@jit
def q_homstat_calc(s,tau,T):
    """
    q^j(s, tau) = exp( -(s - tau) / T )
    (Eq. 8)
    """
    return jnp.exp(-(s - tau) / T)


# ==========================================
# Material - non homeostatic
# ==========================================

def K_exp(constituent, k, sigma_s, sigma_0):
    1/constituent.T*(1+k*(sigma_s-sigma_0)/sigma_s)


def Q_calc(constituent,tau_history):
    """
    Q^j(s) = exp(-s/T)
    (Eq. 8)
    """
    integrand = K_exp(constituent, constituent.k_sigma_plus,constituent.sigma_f,constituent.sigma_f[0])
    return jnp.trapezoid(-integrand, tau_history)


def q_calc(constituent,tau,tau_history):
    """
    q^j(s, tau) = exp( -(s - tau) / T )
    (Eq. 8)
    """
    integrand = K_exp(constituent, constituent.k_sigma_plus[tau_history[>tau]],constituent.sigma_f[tau_history[>tau]],constituent.sigma_f[0])
    return jnp.trapezoid(-integrand, tau_history)




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
 



def W_calc(constituent, F_s, F_tau):
    """W(F_e)."""
    F_e = F_e_calc(F_s, F_tau, jnp.eye(3), jnp.eye(3), constituent.G)
    return constituent.W(F_e)

def Psi_calc(constituent, s, F_s, F_history, tau_history, Q, q):
    '''
    Psi^j(s) = rho_0^j * Q^j(s) * W(F_s, F(0))
               + integral_0^s m^j(tau) * q^j(s,tau) * W(F_s, F(tau)) dtau
    (Eq. 3)
    '''
    tau_history = jnp.asarray(tau_history)
    m_history = jnp.asarray(constituent.m_history)
 
    first_term = constituent.rho_history[0] * Q(s, constituent.T) * \
        W_calc(constituent, F_s, F_history[0])
 
    q_values = q(s, tau_history, constituent.T)
    W_values = jax.vmap(W_calc, in_axes=(None, None, 0))(
        constituent, F_s, F_history
    )
    integrand = m_history * q_values * W_values
    second_term = jnp.trapezoid(integrand, tau_history)
 
    return first_term + second_term




if __name__ =='__main__':
    T = 101.0
    rho_0 = 1.0
    c1, c2 = 10.0, 5.0
    direction = (0.0, 0.0, 1.0)
    G = jnp.eye(3)
    lam_step = 1.05
    ds = 1.0
    n_steps = 50


    W = fung_strain_energy(c1, c2, direction)
    fiber = constituent(rho_0, T, G, W)
 
    F0 = F_calc_uniax(1.0, direction)
    F_s = F_calc_uniax(lam_step, direction)
 
    F_history = [F0]
    tau_history = [0.0]
    s = 0.0
    for step in range(n_steps):
        s += ds
        F_history.append(F_s)
        tau_history.append(s)
        fiber.m_history.append(rho_0 / T)  # homeostatic, constant
 
    F_history_arr = jnp.stack(F_history)
    tau_history_arr = jnp.array(tau_history)
 
    psi_val = Psi_calc(fiber, s, F_s, F_history_arr, tau_history_arr, Q_homstat_calc, q_homstat_calc)
    rho_val = rho_calc(s, fiber, tau_history_arr, Q_homstat_calc, q_homstat_calc)
 
    C = C_calc(F_s)
    I4 = I4_calc(C, direction)
    W0_val = (c1 / (2 * c2)) * (jnp.exp(c2 * (I4 - 1) ** 2) - 1)
    psi_closed_form = rho_0 * jnp.exp(-s / T) * W0_val
 
    print(f'Psi (this file):        {float(psi_val):.6f}')
    print(f'Psi (closed form):      {float(psi_closed_form):.6f}')
    print(f'diff:                   {float(abs(psi_val - psi_closed_form)):.6e}')
    print(f'rho(s) [should be 1.0]: {float(rho_val):.6f}')
