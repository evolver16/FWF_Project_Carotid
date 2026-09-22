import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================


@jit
def F_e_calc(F, F_g, F_r):
    """F_e^j(s,tau) = F(s) F_g(s)^-1 [ F(tau) F_g(tau)^-1 ]^-1 G^j"""
    inner = F @ jnp.linalg.inv(F_g)
    return F @ jnp.linalg.inv(F_g) @ jnp.linalg.inv(inner) @ F_r


def sym_to_voigt(T):
    """(3,3) symmetric -> (6,) in (11,22,33,12,13,23) order."""
    return jnp.array([T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2]])

def voigt_to_sym(v):
    """(6,) -> (3,3) symmetric."""
    return jnp.array([[v[0], v[3], v[4]],
                       [v[3], v[1], v[5]],
                       [v[4], v[5], v[2]]])


@jit
def polar_rotation(F):
    """R from the right polar decomposition F = R U, via SVD (F = U_ S V^T -> R = U_ @ V^T).
    Used once per step (Eq. 17) to rotate the fixed deposition prestress sigma_pre."""
    U_, _, Vt = jnp.linalg.svd(F)
    return U_ @ Vt

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
    def F_r(self, F_e, J, c, ds, sigma_rate_euler):
        """Eq. 41 closed form. sigma_rate_euler = rho_dot_plus/rho * (sigma - sigma_pre),
        built by the caller from THIS step's values (same tensor NeoHookean.F_r's
        Newton solve uses as rhs) — scalarized here via self.sigma_f (trace),
        since it's linear: self.sigma_f(sigma_rate_euler) ==
        rho_dot_plus/rho * (sigma_f - sigma_f_pre)."""
        lam_r = jnp.dot(self.M, c.F_r @ self.M)
        I4 = self.I4(F_e)
        denom = self.d2Psi_dI4(I4) * I4**2 + self.dPsi_dI4(I4) * I4
        lam_rate = self.sigma_f(sigma_rate_euler) * (J * c.phi) / (4 * c.rho * lam_r * denom)
        lam_r_new = lam_r + ds * lam_rate
        return lam_r_new * self.P + (1.0 / jnp.sqrt(lam_r_new)) * (jnp.eye(3) - self.P)


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
    def _residual(self, F_r_s, F_e, F_r, sigma_rate_euler, ds):
        """Eq. 20 residual, discretized via Eq. 43 (Ḟr ≈ (Fr_new - Fr_old)/ds).
        x: (6,) Voigt guess for F_r_new."""
        F_r_s = voigt_to_sym(F_r_s)
        L_r = ((F_r_s - F_r) / ds) @ jnp.linalg.inv(F_r)
        _, dsigma = jax.jvp(self.sigma, (F_e,), (F_e @ L_r,))
        return sym_to_voigt(dsigma - sigma_rate_euler)

    @jit
    def F_r(self, F_e, J, c, ds, sigma_rate_euler):
        """Solve Eq. 20 for F_r_new via Newton's method. Tensor in, tensor out —
        same calling shape as Fung.F_r, iteration fully internal."""
        def newton_step(i, x):
            fvec = self._residual(x, F_e, c.F_r, sigma_rate_euler, ds)
            fjac = jax.jacfwd(self._residual)(x, F_e, c.F_r, sigma_rate_euler, ds)
            return x + jnp.linalg.solve(fjac, -fvec)

        x0 = sym_to_voigt(c.F_r)
        x_final = jax.lax.fori_loop(0, 20, newton_step, x0)
        return voigt_to_sym(x_final)


# ==========================================
# State containers
# ==========================================

@jax.tree_util.register_pytree_node_class
class constituent:
    def __init__(self, material, T, rho_0, k_sigma_plus, k_sigma_minus,
                 G=None, sigma_pre=None, F_r=None, phi=1.0):
        self.material = material
        self.T = T
        self.rho = rho_0
        self.k_sigma_plus = jnp.asarray(k_sigma_plus)
        self.k_sigma_minus = jnp.asarray(k_sigma_minus)

        if sigma_pre is None:
            self.sigma_pre = material.sigma(G)
        else:
            self.sigma_pre = sigma_pre
        self.sigma = self.sigma_pre
        self.sigma_f_pre = material.sigma_f(self.sigma_pre)
        self.sigma_f = self.sigma_f_pre

        if F_r is not None:
            self.F_r = F_r
        elif G is not None:
            self.F_r = jnp.linalg.inv(G)
        else:
            self.F_r = jnp.eye(3)

        self.phi = jnp.asarray(phi)
        self.rho_dot_plus = jnp.zeros(())

    def tree_flatten(self):
        children = (self.material, self.T, self.rho, self.k_sigma_plus,
                    self.k_sigma_minus, self.F_r, self.sigma_pre,
                    self.sigma_f_pre, self.sigma, self.sigma_f, self.phi,
                    self.rho_dot_plus)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        (obj.material, obj.T, obj.rho, obj.k_sigma_plus, obj.k_sigma_minus,
         obj.F_r, obj.sigma_pre, obj.sigma_f_pre, obj.sigma, obj.sigma_f,
         obj.phi, obj.rho_dot_plus) = children
        return obj

@jax.tree_util.register_pytree_node_class
class mixture:
    def __init__(self, constituents, ds, ag=None):
        self.constituents = constituents
        self.ds = ds
        self.rho_tot_0 = sum(c.rho for c in constituents)
        self.rho_tot = self.rho_tot_0

        if ag is None:
            self.F_g_calc = self.F_g_iso_calc
            self.ag = None
            self.F_g = self.F_g_calc(self.ag)
        else:
            ag = ag/jnp.linalg.norm(ag)
            self.F_g_calc = self.F_g_aniso_calc
            self.ag = ag
            self.F_g = self.F_g_calc(self.ag)

    def F_g_iso_calc(self, ag):
        """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 6, isotropic)"""
        return (self.rho_tot/self.rho_tot_0) ** (1.0 / 3.0) * jnp.eye(3)

    def F_g_aniso_calc(self, ag):
        """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 7, anisotropic)"""
        return (self.rho_tot/self.rho_tot_0 - 1) * jnp.outer(ag, ag)  + jnp.eye(3)

    def tree_flatten(self):
        is_iso = self.ag is None
        children = (self.constituents, self.ds, self.rho_tot_0, self.rho_tot,
                    self.F_g, self.ag)
        return children, is_iso

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        (obj.constituents, obj.ds, obj.rho_tot_0, obj.rho_tot,
         obj.F_g, obj.ag) = children
        is_iso = aux
        obj.F_g_calc = obj.F_g_iso_calc if is_iso else obj.F_g_aniso_calc
        return obj




# ==========================================
# Density rate (Eq. 24) and update (Eq. 37)
# ==========================================


def _sigma_f_rel(c, sigma_f, eps):
    """(sigma_f - sigma_f_pre)/sigma_f_pre"""
    denom = jnp.where(jnp.abs(c.sigma_f_pre) < eps, 1.0, c.sigma_f_pre)
    return (sigma_f - c.sigma_f_pre) / denom


@jit
def rho_dot_plus_calc(c, sigma_f, eps=1e-9):
    """Eq. 24: rho_dot_plus = (rho/T)*(1 + k_sigma+ * (sigma_f-sigma_f_pre)/sigma_f_pre)"""
    rel = _sigma_f_rel(c, sigma_f, eps)
    return (c.rho / c.T) * (1.0 + c.k_sigma_plus * rel)


@jit
def rho_dot_minus_calc(c, sigma_f, eps=1e-9):
    """Eq. 24: rho_dot_minus = -(rho/T)*(1 + k_sigma- * (sigma_f-sigma_f_pre)/sigma_f_pre)."""
    rel = _sigma_f_rel(c, sigma_f, eps)
    return -(c.rho / c.T) * (1.0 + c.k_sigma_minus * rel)





# ==========================================
# Per-constituent step
# ==========================================


@jit
def mixture_sigma_solver(mixt,F):
    F_g = mixt.F_g
    J = jnp.linalg.det(F)
    sigma_tot = jnp.zeros((3,3))
    for c in mixt.constituents:
        F_e = F @ jnp.linalg.inv(F_g) @ jnp.linalg.inv(c.F_r)
        sigma_tot += c.material.sigma(F_e) * c.rho / J
    return sigma_tot


@jit
def update_constituents(mixt,F):
    ds = mixt.ds
    F_g = mixt.F_g
    J = jnp.linalg.det(F)
    for c in mixt.constituents:
        R = polar_rotation(F)
        sigma_pre_s = R @ c.sigma_pre @ R.T
        F_e = F @ jnp.linalg.inv(F_g) @ jnp.linalg.inv(c.F_r)
        sigma_s = c.material.sigma(F_e)

        sigma_f_s = c.material.sigma_f(sigma_s)

        rho_dot_plus = rho_dot_plus_calc(c, sigma_f_s)
        rho_dot_minus = rho_dot_minus_calc(c, sigma_f_s)
        rho_s = c.rho + (rho_dot_plus + rho_dot_minus) * ds

        sigma_rate_euler = rho_dot_plus/c.rho*(sigma_s-sigma_pre_s)
        c.material.F_r(F_e, J, c, ds, sigma_rate_euler)