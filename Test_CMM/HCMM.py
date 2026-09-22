"""Homogenized constrained mixture model (HCMM), Maes & Famaey (2023) sec. 2.2.

Each constituent carries one remodeling deformation gradient F_r^j instead of a
cohort history:

    F_e^j(s) = F(s) F_g(s)^-1 F_r^j(s)^-1                          (Eq. 15)
    rho_dot_+-^j(s) = +-(rho^j/T^j)(1 + k_sigma_+- rel)            (Eq. 24)
    rho^j(s+ds)     = rho^j(s) + rho_dot^j(s) ds                   (Eq. 37)
    (rho_dot_+/rho)(sigma^j - sigma_pre^j)
        = (d sigma^j / d F_e^j) : (F_e^j L_r^j)                    (Eq. 20)

Eq. 20 has a closed form for a 1-D fiber family (Eq. 41, Fung) and is solved
numerically for a 3-D compressible matrix (Eq. 43 + Newton, NeoHookean).

Time is carried in step units: T is given in days and divided by ds once
(T_steps = T/ds), so the Euler updates add their increments directly.

Solver entry point (shared with FCMM.py -- see jaxFEM_solver.py):
    sigma_tot, aux = sigma_solver(state, F)
    state          = commit(state, F, aux)
"""

import jax
import jax.numpy as jnp
from jax import jit

# ==========================================
# Kinematics
# ==========================================


@jit
def F_e_calc(F, F_g, F_r):
    """F_e^j(s) = F(s) F_g(s)^-1 F_r^j(s)^-1                        (Eq. 15)"""
    return F @ jnp.linalg.inv(F_g) @ jnp.linalg.inv(F_r)


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
    """R from the polar decomposition F = R U, via SVD F = U_ S V^T -> R = U_ V^T.

    R(s) rotates the deposition prestress: sigma_pre^j(s) = R sigma^j(0) R^T
    (Eq. 17).
    """
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
        M = jnp.asarray(M)
        self.M = M / jnp.linalg.norm(M)
        self.P = jnp.outer(self.M, self.M)

    def tree_flatten(self):
        return (self.k1, self.k2, self.M), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def I4(self, F):
        """I4 = M . (F^T F) M = |F M|^2, the squared fiber stretch  (Eq. 28)"""
        FM = F @ self.M
        return jnp.dot(FM, FM)

    @jit
    def Psi_I4(self, I4):
        """W^coll(I4) = k1/(2 k2) [exp(k2 (I4 - 1)^2) - 1]   for I4 > 1
                    = 0                                     otherwise  (Eq. 28)

        Fibers carry load in tension only (Epos in sigma_C_HGO_C_fiber.for).
        """
        return jnp.where(
            I4 > 1.0,
            (self.k1 / (2 * self.k2)) * (jnp.exp(self.k2 * (I4 - 1) ** 2) - 1),
            0.0,
        )

    @jit
    def dPsi_dI4(self, I4):
        """dW^coll/dI4"""
        return jax.grad(self.Psi_I4)(I4)

    @jit
    def d2Psi_dI4(self, I4):
        """d^2 W^coll/dI4^2"""
        return jax.grad(self.dPsi_dI4)(I4)

    @jit
    def Psi_F(self, F):
        """W^coll as a function of F, via I4(F)                        (Eq. 28)"""
        return self.Psi_I4(self.I4(F))

    @jit
    def sigma(self, F):
        """sigma = (dW/dF) F^T = 2 (dW/dI4) (F M) (x) (F M)           (Eq. 29)"""
        dW_dF = jax.grad(self.Psi_F)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = I_1(sigma) = tr(sigma)                            (Eq. 31)"""
        return jnp.trace(sigma)

    @jit
    def F_r(self, F_e, J, c, sigma_rate_euler):
        """Closed-form Eq. 20 for a 1-D fiber family:

        lam_r_dot = (rho_dot_+/rho)(sigma_f - sigma_pre_f) (J phi / 4 rho) lam_r
                    [ (d^2W/dI4^2) I4^2 + (dW/dI4) I4 ]^-1              (Eq. 41)

        F_r = lam_r M(x)M + lam_r^-1/2 (I - M(x)M)                      (Eq. 38)
        """
        lam_r = jnp.dot(self.M, c.F_r @ self.M)
        I4 = self.I4(F_e)
        denom = self.d2Psi_dI4(I4) * I4**2 + self.dPsi_dI4(I4) * I4
        slack = jnp.abs(denom) < 1e-12
        denom_safe = jnp.where(slack, 1.0, denom)
        lam_rate = jnp.where(
            slack, 0.0,
            self.sigma_f(sigma_rate_euler) * (J * c.phi) / (4 * c.rho) * lam_r / denom_safe)
        lam_r_new = lam_r + lam_rate
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
        """W^elas = C10 (I1bar - 3) + K/2 (J_e - 1)^2                  (Eq. 28)

        with I1bar = J_e^(-2/3) tr(F^T F) and J_e = det F_e.
        """
        I1 = jnp.trace(F.T @ F)
        J = jnp.linalg.det(F)
        I1_inc = I1 * J ** (-2 / 3)
        return self.C10 * (I1_inc - 3) + self.K / 2 * (J - 1) ** 2

    @jit
    def sigma(self, F):
        """sigma = (dW/dF) F^T                                         (Eq. 29)

        The rho^j/(phi^j J) prefactor is applied by the mixture, not here.
        """
        dW_dF = jax.grad(self.Psi)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = I_1(sigma) = tr(sigma)                            (Eq. 31)"""
        return jnp.trace(sigma)

    @jit
    def _residual(self, F_r_s, F_e, F_r, sigma_rate_euler):
        """Residual of Eq. 20, with F_r_dot discretized by Eq. 43:

        r(F_r(s+ds)) = (d sigma/d F_e) : (F_e L_r)
                       - (rho_dot_+/rho)(sigma - sigma_pre)             (Eq. 20)
        L_r = F_r_dot F_r^-1,  F_r_dot = [F_r(s+ds) - F_r(s)]/ds        (Eq. 43)
        """
        F_r_s = voigt_to_sym(F_r_s)
        L_r = (F_r_s - F_r) @ jnp.linalg.inv(F_r)
        _, dsigma = jax.jvp(self.sigma, (F_e,), (F_e @ L_r,))
        return sym_to_voigt(dsigma - sigma_rate_euler)

    @jit
    def F_r(self, F_e, J, c, sigma_rate_euler):
        """F_r(s+ds) solving Eq. 20, by Newton's method on the six independent
        components of F_r."""
        def newton_step(i, x):
            fvec = self._residual(x, F_e, c.F_r, sigma_rate_euler)
            fjac = jax.jacfwd(self._residual)(x, F_e, c.F_r, sigma_rate_euler)
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
        """F_g(s) = [rho_tot(s)/rho_tot(0) - 1] a_g (x) a_g + I     (Eq. 7)"""
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
# Mass production and removal (Eq. 24)
# ==========================================


def _sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0, eps):
    """Mechanobiological driver of Eq. 24:

        rel = [rho_tot sigma_f(s) - rho_tot(0) sigma_f(0)]
              / [rho_tot(0) sigma_f(0)]

    sigma_f carries a 1/J, the setpoint sigma_f(0) does not
    (UMAT_GR_HCM.for: s_fib = 2/det dW I4 against s_fib_hom = 2 dW_hom g^2).
    The denominator may be omitted when sigma_f(0) = 0 (Eq. 24).
    """
    ref = rho_tot_0 * c.sigma_f_pre
    denom = jnp.where(jnp.abs(ref) < eps, 1.0, ref)
    return (rho_tot * sigma_f - ref) / denom


@jit
def rho_dot_plus_calc(c, sigma_f, ds, rho_tot, rho_tot_0, eps=1e-9):
    """rho_dot_+ = (rho/T)(1 + k_sigma_+ rel) ds                     (Eq. 24)
    """
    rel = _sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0, eps)
    return (c.rho / (c.T / ds)) * (1.0 + c.k_sigma_plus * rel)


@jit
def rho_dot_minus_calc(c, sigma_f, ds, rho_tot, rho_tot_0, eps=1e-9):
    """rho_dot_- = -(rho/T)(1 + k_sigma_- rel) ds                    (Eq. 24)"""
    rel = _sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0, eps)
    return -(c.rho / (c.T / ds)) * (1.0 + c.k_sigma_minus * rel)





# ==========================================
# Per-constituent step
# ==========================================


@jit
def mixture_sigma_solver(mixt, F):
    """sigma_tot = sum_j phi^j sigma^j                                (Eq. 27)

    with sigma^j = rho^j/(phi^j J) (dW^j/dF_e^j) F_e^jT (Eq. 17), so phi^j
    cancels and each constituent contributes rho^j/J times its material stress.

    Also returns the per-constituent (F_e, sigma) for commit to reuse.
    """
    F_g = mixt.F_g
    J = jnp.linalg.det(F)
    sigma_tot = jnp.zeros((3,3))
    trial = []
    for c in mixt.constituents:
        F_e = F_e_calc(F, F_g, c.F_r)
        sigma = c.material.sigma(F_e)
        sigma_tot += sigma * c.rho / J
        trial.append((F_e, sigma))
    return sigma_tot, trial


@jit
def constituent_update(c, F_e, sigma_s, R, ds, J, rho_tot, rho_tot_0):
    """One constituent's growth and remodeling step at a settled F:

        sigma_pre^j(s) = R sigma^j(0) R^T                             (Eq. 17)
        sigma_f^j(s)   = tr(sigma^j)/J                                (Eq. 31)
        rho^j(s+ds)    = rho^j + (rho_dot_+ + rho_dot_-)              (Eq. 24/37)
        F_r^j(s+ds)    from Eq. 20, closed form (Eq. 41) or Newton (Eq. 43)

    The Eq. 20 source is density-weighted the same way as the Eq. 24 driver.
    F_e and sigma_s come from mixture_sigma_solver; no stress is re-evaluated.
    """
    sigma_pre_s = R @ c.sigma_pre @ R.T
    sigma_f_s = c.material.sigma_f(sigma_s) / J

    rho_dot_plus = rho_dot_plus_calc(c, sigma_f_s, ds, rho_tot, rho_tot_0)
    rho_dot_minus = rho_dot_minus_calc(c, sigma_f_s, ds, rho_tot, rho_tot_0)
    rho_s = c.rho + rho_dot_plus + rho_dot_minus

    sigma_rate_euler = (rho_dot_plus / c.rho) * (rho_tot * sigma_s / J
                                                 - rho_tot_0 * sigma_pre_s)
    F_r_s = c.material.F_r(F_e, J, c, sigma_rate_euler)
    return rho_s, F_r_s, rho_dot_plus


# ==========================================
# Solver entry point (shared with FCMM.py -- see jaxFEM_solver.py)
#   sigma_solver(state, F) -> (sigma_tot, aux)   pure, committed state only
#   commit(state, F, aux)  -> state              once, on the converged F
# ==========================================


# Trial stress at F from committed state only; no F_r solve. aux carries the
# per-constituent (F_e, sigma) that commit reuses.
sigma_solver = mixture_sigma_solver


def commit(mixt, F, aux):
    """Advance the mixture one step on the settled F:

        rho^j, F_r^j  per constituent                                (Eq. 24/37, 20)
        rho_tot(s)    = sum_j rho^j(s)                               (Eq. 25)
        phi^j(s)      = rho^j(s)/rho_tot(s)
        F_g(s)        from rho_tot(s)                                (Eq. 6/7)

    F_g is written for the NEXT step; this step used the one it came in with.
    Not jitted: it mutates the state containers, and a mutation inside a traced
    function would only take effect at trace time.
    """
    ds, J = mixt.ds, jnp.linalg.det(F)
    R = polar_rotation(F)
    updates = [constituent_update(c, F_e, sigma_s, R, ds, J,
                                  mixt.rho_tot, mixt.rho_tot_0)
               for c, (F_e, sigma_s) in zip(mixt.constituents, aux)]

    rho_tot_s = sum(u[0] for u in updates)
    for c, (rho_s, F_r_s, rho_dot_plus) in zip(mixt.constituents, updates):
        c.rho = rho_s
        c.F_r = F_r_s
        c.rho_dot_plus = rho_dot_plus
        c.phi = rho_s / rho_tot_s

    # F_g_calc reads mixt.rho_tot, so the new total has to land first. The F_g
    # written here is next step's -- this step used the one it came in with.
    mixt.rho_tot = rho_tot_s
    mixt.F_g = mixt.F_g_calc(mixt.ag)
    return mixt