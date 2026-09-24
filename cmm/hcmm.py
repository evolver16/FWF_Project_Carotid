"""Homogenized constrained mixture model (HCMM), Maes & Famaey (2023) sec. 2.2.

    F_e^j(s)        = F(s) F_g(s)^-1 F_r^j(s)^-1                    (Eq. 15)
    rho_dot_+-^j(s) = +-(rho^j/T^j)(1 + k_sigma_+- rel)              (Eq. 24)
    rho^j(s+ds)     = rho^j(s) + rho_dot^j(s) ds                     (Eq. 37)
    (rho_dot_+/rho)(sigma^j - sigma_pre^j)
                    = (d sigma^j / d F_e^j) : (F_e^j L_r^j)          (Eq. 20)

Solver interface: sigma_tot, aux = sigma_solver(state, F); state = commit(state, F, aux)
"""

import jax
import jax.numpy as jnp
from jax import jit
from tensor3 import det3, inv3


@jit
def F_e_calc(F, F_g, F_r):
    """F_e^j(s) = F(s) F_g(s)^-1 F_r^j(s)^-1   (Eq. 15)"""
    return F @ inv3(F_g) @ inv3(F_r)


def sym_to_voigt(T):
    """(3,3) -> (11,22,33,12,13,23)"""
    return jnp.array([T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2]])

def voigt_to_sym(v):
    """(11,22,33,12,13,23) -> (3,3)"""
    return jnp.array([[v[0], v[3], v[4]],
                       [v[3], v[1], v[5]],
                       [v[4], v[5], v[2]]])


@jit
def polar_rotation(F, n_iter=12):
    """F = R U,  R_{k+1} = (R_k + R_k^-T)/2 from R_0 = F, smooth at repeated singular values"""
    return jax.lax.fori_loop(0, n_iter, lambda _, R: 0.5 * (R + inv3(R).T), F)


@jax.tree_util.register_pytree_node_class
class Fung:
    def __init__(self, k1, k2, M):
        self.k1 = k1
        self.k2 = k2
        M = jnp.asarray(M)
        self.M = M / jnp.linalg.norm(M)

    @property
    def P(self):
        return jnp.outer(self.M, self.M)

    def tree_flatten(self):
        return (self.k1, self.k2, self.M), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.k1, obj.k2, obj.M = children
        return obj

    @jit
    def I4(self, F):
        """I4 = |F M|^2   (Eq. 28)"""
        FM = F @ self.M
        return jnp.dot(FM, FM)

    @jit
    def Psi_I4(self, I4):
        """W^coll = k1/(2 k2) [exp(k2 (I4 - 1)^2) - 1] if I4 > 1 else 0   (Eq. 28)"""
        return jnp.where(
            I4 > 1.0,
            (self.k1 / (2 * self.k2)) * (jnp.exp(self.k2 * (I4 - 1) ** 2) - 1),
            0.0,
        )

    @jit
    def dPsi_dI4(self, I4):
        return jax.grad(self.Psi_I4)(I4)

    @jit
    def d2Psi_dI4(self, I4):
        return jax.grad(self.dPsi_dI4)(I4)

    @jit
    def Psi_F(self, F):
        return self.Psi_I4(self.I4(F))

    @jit
    def sigma(self, F):
        """sigma = (dW/dF) F^T = 2 (dW/dI4) (F M)(x)(F M)   (Eq. 29)"""
        dW_dF = jax.grad(self.Psi_F)(F)
        return dW_dF @ F.T

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = tr(sigma)   (Eq. 31)"""
        return jnp.trace(sigma)

    @jit
    def F_r(self, F_e, J, c, sigma_rate_euler):
        """lam_r_dot = (rho_dot_+/rho)(sigma_f - sigma_pre_f) (J phi / 4 rho) lam_r
                       [(d^2W/dI4^2) I4^2 + (dW/dI4) I4]^-1                     (Eq. 41)
        F_r = lam_r M(x)M + lam_r^-1/2 (I - M(x)M)                              (Eq. 38)
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
        """W^elas = C10 (J_e^(-2/3) tr(F^T F) - 3) + K/2 (J_e - 1)^2   (Eq. 28)"""
        I1 = jnp.trace(F.T @ F)
        J = det3(F)
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

    @jit
    def _residual(self, F_r_s, F_e, F_r, sigma_rate_euler, scale):
        """r = rho/(phi J) (d sigma/d F_e) : (F_e L_r) - (rho_dot_+/rho)(sigma - sigma_pre)   (Eq. 20)
        L_r = [F_r(s+ds) - F_r(s)] F_r^-1                                                    (Eq. 43)
        """
        F_r_s = voigt_to_sym(F_r_s)
        L_r = (F_r_s - F_r) @ inv3(F_r)
        _, dsigma = jax.jvp(self.sigma, (F_e,), (F_e @ L_r,))
        return sym_to_voigt(scale * dsigma - sigma_rate_euler)

    @jit
    def F_r(self, F_e, J, c, sigma_rate_euler):
        """Symmetric F_r(s+ds) from one linear solve, Eq. 20 is affine in F_r(s+ds)."""
        scale = c.rho / (c.phi * J)
        r = lambda x: self._residual(x, F_e, c.F_r, sigma_rate_euler, scale)
        x0 = sym_to_voigt(c.F_r)
        return voigt_to_sym(x0 - jnp.linalg.solve(jax.jacfwd(r)(x0), r(x0)))


@jax.tree_util.register_pytree_node_class
class NeoHookeanInc:
    def __init__(self, C10):
        self.C10 = C10

    def tree_flatten(self):
        return (self.C10,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def Psi(self, F):
        """W^elas = C10 (tr(F^T F) - 3)   (Eq. 32)"""
        return self.C10 * (jnp.trace(F.T @ F) - 3)

    @jit
    def sigma(self, F):
        """sigma = 2 C10 (B - I1/3 I), B = F F^T; pressure added by the solver   (Eq. 33)"""
        B = F @ F.T
        return 2 * self.C10 * (B - jnp.trace(B) / 3 * jnp.eye(3))

    @staticmethod
    @jit
    def sigma_f(sigma):
        """sigma_f = tr(sigma)   (Eq. 31)"""
        return jnp.trace(sigma)

    @jit
    def _residual(self, F_r_s, F_e, F_r, sigma_rate_euler, scale):
        """r = dev[rho/(phi J) (d sigma/d F_e) : (F_e L_r) - (rho_dot_+/rho)(sigma - sigma_pre)]   (Eq. 20, 44)
        r_6 = det F_r(s+ds) - 1
        """
        F_r_s = voigt_to_sym(F_r_s)
        L_r = (F_r_s - F_r) @ inv3(F_r)
        _, dsigma = jax.jvp(self.sigma, (F_e,), (F_e @ L_r,))
        r = sym_to_voigt(scale * dsigma - sigma_rate_euler)
        return jnp.concatenate([r[jnp.array([0, 1, 3, 4, 5])],
                                det3(F_r_s)[None] - 1.0])

    @jit
    def F_r(self, F_e, J, c, sigma_rate_euler, tol=1e-14, max_iter=20):
        """Newton on symmetric F_r(s+ds) with det F_r = 1; implicit gradient via custom_root."""
        scale = c.rho / (c.phi * J)
        r = lambda x: self._residual(x, F_e, c.F_r, sigma_rate_euler, scale)

        def newton(f, x0):
            def body(state):
                x, _, i = state
                x = x - jnp.linalg.solve(jax.jacfwd(f)(x), f(x))
                return x, jnp.linalg.norm(f(x)), i + 1

            return jax.lax.while_loop(lambda s: (s[1] > tol) & (s[2] < max_iter), body,
                                      (x0, jnp.linalg.norm(f(x0)), 0))[0]

        x = jax.lax.custom_root(r, sym_to_voigt(c.F_r), newton,
                                lambda g, y: jnp.linalg.solve(jax.jacobian(g)(y), y))
        return voigt_to_sym(x)


def _concrete_inf(T):
    """True only for a concrete T = inf (no turnover); traced T keeps the general path."""
    try:
        return bool(jnp.isinf(T))
    except jax.errors.ConcretizationTypeError:
        return False


@jax.tree_util.register_pytree_node_class
class constituent:
    def __init__(self, material, T, rho_0, k_sigma_plus, k_sigma_minus,
                 G=None, sigma_pre=None, F_r=None, phi=1.0,
                 sigma_pre_mode="deposition"):
        """sigma_pre_mode: "deposition" (F_e -> G, Cyron/CMM) or "initial" (Maes Eq. 17)."""
        if sigma_pre_mode not in ("deposition", "initial"):
            raise ValueError(f"sigma_pre_mode must be 'deposition' or 'initial', got {sigma_pre_mode!r}")
        self.sigma_pre_mode = sigma_pre_mode
        self.turnover = not _concrete_inf(T)
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
            self.F_r = inv3(G)
        else:
            self.F_r = jnp.eye(3)

        self.phi = jnp.asarray(phi)
        self.rho_dot_plus = jnp.zeros(())

    def tree_flatten(self):
        children = (self.material, self.T, self.rho, self.k_sigma_plus,
                    self.k_sigma_minus, self.F_r, self.sigma_pre,
                    self.sigma_f_pre, self.sigma, self.sigma_f, self.phi,
                    self.rho_dot_plus)
        return children, (self.sigma_pre_mode, self.turnover)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        (obj.material, obj.T, obj.rho, obj.k_sigma_plus, obj.k_sigma_minus,
         obj.F_r, obj.sigma_pre, obj.sigma_f_pre, obj.sigma, obj.sigma_f,
         obj.phi, obj.rho_dot_plus) = children
        obj.sigma_pre_mode, obj.turnover = aux
        return obj

    def replace(self, **fields):
        children, aux = self.tree_flatten()
        obj = constituent.tree_unflatten(aux, children)
        for k, v in fields.items():
            setattr(obj, k, v)
        return obj

@jax.tree_util.register_pytree_node_class
class mixture:
    def __init__(self, constituents, ds, ag=None):
        self.constituents = constituents
        self.ds = ds
        self.rho_tot_0 = sum(c.rho for c in constituents)
        self.rho_tot = self.rho_tot_0
        for c in constituents:
            c.phi = c.rho / self.rho_tot_0

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
        """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 6)"""
        return (self.rho_tot/self.rho_tot_0) ** (1.0 / 3.0) * jnp.eye(3)

    def F_g_aniso_calc(self, ag):
        """F_g(s) = (rho_tot(s)/rho_tot(0) - 1) ag(x)ag + I   (Eq. 7)"""
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

    def replace(self, **fields):
        children, aux = self.tree_flatten()
        obj = mixture.tree_unflatten(aux, children)
        for k, v in fields.items():
            setattr(obj, k, v)
        return obj


def _sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0, eps):
    """rel = [rho_tot sigma_f(s) - rho_tot(0) sigma_f(0)] / [rho_tot(0) sigma_f(0)]   (Eq. 24)

    Denominator dropped when sigma_f(0) = 0.
    """
    ref = rho_tot_0 * c.sigma_f_pre
    denom = jnp.where(jnp.abs(ref) < eps, 1.0, ref)
    return (rho_tot * sigma_f - ref) / denom


@jit
def rho_dot_plus_calc(c, sigma_f, ds, rho_tot, rho_tot_0, eps=1e-9):
    """rho_dot_+ ds = (rho/T)(1 + k_sigma_+ rel) ds   (Eq. 24)"""
    rel = _sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0, eps)
    return (c.rho / (c.T / ds)) * (1.0 + c.k_sigma_plus * rel)


@jit
def rho_dot_minus_calc(c, sigma_f, ds, rho_tot, rho_tot_0, eps=1e-9):
    """rho_dot_- ds = -(rho/T)(1 + k_sigma_- rel) ds   (Eq. 24)"""
    rel = _sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0, eps)
    return -(c.rho / (c.T / ds)) * (1.0 + c.k_sigma_minus * rel)


@jit
def mixture_sigma_solver(mixt, F):
    """sigma_tot = sum_j phi^j sigma^j = sum_j (rho^j/J) sigma_mat^j(F_e^j)   (Eq. 17, 27)"""
    F_g = mixt.F_g
    J = det3(F)
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
    """sigma_pre^j(s) = (J_g/J) R sigma^j(0) R^T          deposition (Cyron Eq. 11)
                   = R sigma^j(0) R^T                   initial    (Eq. 17)
    sigma_f^j(s)   = tr(sigma^j)/J                      (Eq. 31)
    rho^j(s+ds)    = rho^j + rho_dot_+ + rho_dot_-      (Eq. 24, 37)
    F_r^j(s+ds)    from Eq. 20 via Eq. 41 or Eq. 43
    """
    sigma_f_s = c.material.sigma_f(sigma_s) / J
    if not c.turnover:
        return c.rho, c.F_r, jnp.zeros_like(c.rho_dot_plus), sigma_f_s
    sigma_pre_s = R @ c.sigma_pre @ R.T

    rho_dot_plus = rho_dot_plus_calc(c, sigma_f_s, ds, rho_tot, rho_tot_0)
    rho_dot_minus = rho_dot_minus_calc(c, sigma_f_s, ds, rho_tot, rho_tot_0)
    rho_s = c.rho + rho_dot_plus + rho_dot_minus

    rho_pre = rho_tot / J if c.sigma_pre_mode == "deposition" else rho_tot_0
    sigma_rate_euler = (rho_dot_plus / c.rho) * (rho_tot * sigma_s / J
                                                 - rho_pre * sigma_pre_s)
    F_r_s = c.material.F_r(F_e, J, c, sigma_rate_euler)
    return rho_s, F_r_s, rho_dot_plus, sigma_f_s


sigma_solver = mixture_sigma_solver


def J_target(mixt):
    """J = det F_g for incompressible constituents (det F_e = det F_r = 1)"""
    return det3(mixt.F_g)


@jit
def commit(mixt, F, aux):
    """New state: rho^j, F_r^j, phi^j = rho^j/rho_tot, F_g (Eq. 6/7) on the settled F."""
    J = det3(F)
    R = polar_rotation(F)
    updates = [constituent_update(c, F_e, sigma_s, R, mixt.ds, J, mixt.rho_tot, mixt.rho_tot_0)
               for c, (F_e, sigma_s) in zip(mixt.constituents, aux)]
    rho_tot_s = sum(u[0] for u in updates)
    constituents = [c.replace(rho=rho_s, F_r=F_r_s, rho_dot_plus=rho_dot_plus,
                              phi=rho_s / rho_tot_s, sigma=sigma_s, sigma_f=sigma_f_s)
                    for c, (rho_s, F_r_s, rho_dot_plus, sigma_f_s), (_, sigma_s)
                    in zip(mixt.constituents, updates, aux)]
    new = mixt.replace(constituents=constituents, rho_tot=rho_tot_s)
    return new.replace(F_g=new.F_g_calc(new.ag))
