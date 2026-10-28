"""Removal laws: K(sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0, dsig) -> removal rate K_-^j per unit mass and step"""

import jax.numpy as jnp

from fem.pytree import pytree
from fcmm.core import sigma_f_rel


@pytree(("T", "k"))
class MaesRemoval:
    def __init__(self, T, k):
        self.T, self.k = jnp.asarray(T), jnp.asarray(k)

    def K(self, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0, dsig):
        """K_-^j(s) = (1/T^j)(1 + k_sigma_- rel)   (Eq. 13)"""
        return (1.0 / (self.T / ds)) * (1 + self.k * sigma_f_rel(sigma_f, sigma_f_0, rho_tot, rho_tot_0))


@pytree(("T", "k_turn"))
class MixtureRemoval:
    """first-order removal with the stress-dependent turnover rate of Latorre & Humphrey 2018 (Eq. 10, 11)"""
    def __init__(self, T, k_turn=0.0):
        self.T, self.k_turn = T, k_turn

    def K(self, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0, dsig):
        """K_-^j(s) = (1/T)(1 + k_turn dsig^2)"""
        return (1.0 / (self.T / ds)) * (1.0 + self.k_turn * dsig ** 2)
