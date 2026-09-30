"""Removal laws: K(sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0) -> removal rate K_-^j per unit mass and step"""

import jax.numpy as jnp

from fcmm.core import pytree, sigma_f_rel


@pytree(("T", "k"))
class MaesRemoval:
    def __init__(self, T, k):
        self.T, self.k = jnp.asarray(T), jnp.asarray(k)

    def K(self, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0):
        """K_-^j(s) = (1/T^j)(1 + k_sigma_- rel)   (Eq. 13)"""
        return (1.0 / (self.T / ds)) * (1 + self.k * sigma_f_rel(sigma_f, sigma_f_0, rho_tot, rho_tot_0))
