"""Incompressible neo-Hookean (elastin, hybrid element), Eq. 32"""

import jax
import jax.numpy as jnp
from jax import jit

from materials.voigt import sym_to_voigt, voigt_to_sym
from fem.tensor3 import det3, inv3


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
