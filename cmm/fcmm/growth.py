"""Growth laws: F_g of the mixture from J_g = rho_tot(s) / rho_tot(0)"""

import jax.numpy as jnp

from fcmm.core import pytree


@pytree(())
class Isotropic:
    def F_g(self, J_g):
        """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 6)"""
        return J_g ** (1.0 / 3.0) * jnp.eye(3)


@pytree(("a",))
class Anisotropic:
    def __init__(self, a):
        """growth direction a (normalized), e.g. e_r: thickening"""
        a = jnp.asarray(a)
        self.a = a / jnp.linalg.norm(a)

    def F_g(self, J_g):
        """F_g(s) = (rho_tot(s)/rho_tot(0) - 1) a(x)a + I   (Eq. 7)"""
        return (J_g - 1) * jnp.outer(self.a, self.a) + jnp.eye(3)
