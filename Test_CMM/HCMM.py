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

def sym_to_voigt(T):
    """(3,3) symmetric -> (6,) in (11,22,33,12,13,23) order."""
    return jnp.array([T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2]])

def voigt_to_sym(v):
    """(6,) -> (3,3) symmetric."""
    return jnp.array([[v[0], v[3], v[4]],
                       [v[3], v[1], v[5]],
                       [v[4], v[5], v[2]]])

# ==========================================
# Material
# ==========================================


@jax.tree_util.register_pytree_node_class
class Fung:
    def __init__(self, k1, k2, M, lam_r=1.0):
        self.k1 = k1
        self.k2 = k2
        M = jnp.asarray(M, dtype=jnp.float32)
        self.M = M / jnp.linalg.norm(M)
        self.P = jnp.outer(self.M, self.M)

    def tree_flatten(self):
        return (self.k1, self.k2, self.M), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def I4(self, F):        
        FM = F @ self.M
        return jnp.dot(FM, FM)

    @jit
    def Psi_I4(self, I4):
        return (self.k1 / (2 * self.k2)) * (jnp.exp(self.k2 * (I4 - 1) ** 2) - 1)

    @jit
    def dPsi_dI4(self, I4):
        return jax.grad(self.Psi_I4)(I4)

    @jit
    def d2Psi_dI4(self, I4):
        return jax.grad(self.dPsi_dI4)(I4)

    @jit
    def Psi_F(self, F):
        """W(I4) = k1/(2*k2) * (exp(k2*(I4-1)**2) - 1)"""
        return self.Psi_I4(self.I4(F))

    @jit
    def sigma(self, F):
        dW_dF = jax.grad(self.Psi_F)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        return jnp.trace(sigma)

    @jit
    def lam_rate(self, F_e, F_r, J, c):
        lam_r = jnp.dot(self.M, F_r @ self.M)
        term1 = (c.rho_plus_dt / c.rho) * (c.sigma_f - c.sigma_f_pre)
        term2 = (J * c.phi) / (4 * c.rho * lam_r)
        I4 = self.I4(F_e)
        term3 = 1.0 / (self.d2Psi_dI4(I4) * I4**2 + self.dPsi_dI4(I4) * I4)
        return term1 * term2 * term3


    @jit
    def F_r(self, F, J, c, ds):
        rate = self.lam_rate(F, J, c)
        lam_r_s = self.lam_r + rate * ds
        return lam_r_s * self.P + (1.0/jnp.sqrt(lam_r_s)) * (jnp.eye(3) - self.P)


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

    @jit
    def _residual(self, x, F_e_old, F_r_old, rhs, ds):
        """Eq. 20 residual, discretized via Eq. 43 (Ḟr ≈ (Fr_new - Fr_old)/ds).
        x: (6,) Voigt guess for F_r_new."""
        F_r_new = voigt_to_sym(x)
        L_r = ((F_r_new - F_r_old) / ds) @ jnp.linalg.inv(F_r_old)
        _, dsigma = jax.jvp(self.sigma, (F_e_old,), (F_e_old @ L_r,))
        return sym_to_voigt(dsigma - rhs)

    @jit
    def F_r(self, F_e_old, sigma_old, sigma_pre, rho_dot_plus, rho, F_r_old, ds, n_iter=20):
        """Solve Eq. 20 for F_r_new via Newton's method. Tensor in, tensor out —
        same calling shape as Fung.F_r, iteration fully internal."""
        rhs = (rho_dot_plus / rho) * (sigma_old - sigma_pre)

        def newton_step(i, x):
            fvec = self._residual(x, F_e_old, F_r_old, rhs, ds)
            fjac = jax.jacfwd(self._residual)(x, F_e_old, F_r_old, rhs, ds)
            return x + jnp.linalg.solve(fjac, -fvec)

        x0 = sym_to_voigt(F_r_old)
        x_final = jax.lax.fori_loop(0, n_iter, newton_step, x0)
        return voigt_to_sym(x_final)


# ==========================================
# State containers
# ==========================================

@jax.tree_util.register_pytree_node_class
class constituent:
    def __init__(self, material, T, rho_0, k_sigma_plus, k_sigma_minus,
                 G=None, sigma_pre=None, F_r=None):
        self.material = material
        self.T = T
        self.rho = rho_0
        self.k_sigma_plus = jnp.asarray(k_sigma_plus)
        self.k_sigma_minus = jnp.asarray(k_sigma_minus)
        self.F_r = jnp.eye(3) if F_r is None else F_r

        if sigma_pre is None:
            self.sigma_pre = material.sigma(G)
        else:
            self.sigma_pre = sigma_pre
        self.sigma = self.sigma_pre
        self.sigma_f_pre = material.sigma_f(self.sigma_pre)
        self.sigma_f = self.sigma_f_pre

    def tree_flatten(self):
        children = (self.material, self.T, self.rho, self.k_sigma_plus,
                    self.k_sigma_minus, self.F_r, self.sigma_pre,
                    self.sigma_f_0, self.sigma)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        (obj.material, obj.T, obj.rho, obj.k_sigma_plus, obj.k_sigma_minus,
         obj.F_r, obj.sigma_pre, obj.sigma_f_0, obj.sigma) = children
        return obj



# ==========================================
# Density and volume fractions
# ==========================================




