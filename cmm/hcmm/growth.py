"""Growth laws: F_g of the mixture from the mass ratio rho_tot(s) / rho_tot(0); constituent laws: F_g^j from the
mixture's F_g (constituent(growth=None): F_g^j = F_g)"""

import jax.numpy as jnp

from fem.pytree import pytree


@pytree(())
class Isotropic:
    def F_g(self, ratio):
        """F_g(s) = (rho_tot(s)/rho_tot(0))^(1/3) I   (Eq. 6)"""
        return ratio ** (1.0 / 3.0) * jnp.eye(3)


@pytree(("a",))
class Anisotropic:
    def __init__(self, a):
        """growth direction a (normalized), e.g. e_r: thickening"""
        self.a = a / jnp.linalg.norm(a)

    def F_g(self, ratio):
        """F_g(s) = (rho_tot(s)/rho_tot(0) - 1) a(x)a + I   (Eq. 7)"""
        return (ratio - 1) * jnp.outer(self.a, self.a) + jnp.eye(3)


@pytree(())
class Fixed:
    def of(self, F_g):
        """F_g^j = I: stress-free state fixed, the constituent is only embedded in the growing mixture (no deposition,
        e.g. elastin; full constrained mixture: F_e = F G)"""
        return jnp.eye(3)
