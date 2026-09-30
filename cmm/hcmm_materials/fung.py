"""Collagen fiber family, W = k1/(2 k2) [exp(k2 (I4 - 1)^2) - 1] (tension only)"""

import jax
import jax.numpy as jnp
from jax import jit


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
