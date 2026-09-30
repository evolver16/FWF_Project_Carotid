"""Compressible neo-Hookean (elastin), Eq. 28"""

import jax
import jax.numpy as jnp
from jax import jit

from materials.voigt import sym_to_voigt, voigt_to_sym
from fem.tensor3 import det3, inv3


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
