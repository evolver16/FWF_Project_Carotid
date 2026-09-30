"""Production laws: m(rho, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0) -> mass production m^j per step"""

import jax.numpy as jnp

from fcmm.core import pytree, sigma_f_rel


@pytree(("T", "k"))
class MaesProduction:
    def __init__(self, T, k):
        self.T, self.k = jnp.asarray(T), jnp.asarray(k)

    def m(self, rho, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0):
        """m^j(s) = (rho^j(s)/T^j)(1 + k_sigma_+ rel)   (Eq. 11)"""
        return (rho / (self.T / ds)) * (1 + self.k * sigma_f_rel(sigma_f, sigma_f_0, rho_tot, rho_tot_0))
