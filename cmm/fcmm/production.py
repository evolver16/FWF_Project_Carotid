"""Production laws: m(rho, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0, dsig) -> mass production m^j per step,
dsig = sigma_w / sigma_w_h - 1 (wall stress stimulus, 0 without input)"""

import jax.numpy as jnp

from fem.pytree import pytree
from fcmm.core import sigma_f_rel


@pytree(("T", "k"))
class MaesProduction:
    def __init__(self, T, k):
        self.T, self.k = jnp.asarray(T), jnp.asarray(k)

    def m(self, rho, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0, dsig):
        """m^j(s) = (rho^j(s)/T^j)(1 + k_sigma_+ rel)   (Eq. 11)"""
        return (rho / (self.T / ds)) * (1 + self.k * sigma_f_rel(sigma_f, sigma_f_0, rho_tot, rho_tot_0))


@pytree(("T", "k", "k_turn"))
class MixtureProduction:
    """stimulus from the wall stress (Latorre & Humphrey 2018), as hcmm.MixtureProduction without WSS"""
    def __init__(self, T, k, k_turn=0.0):
        self.T, self.k, self.k_turn = T, k, k_turn

    def m(self, rho, sigma_f, sigma_f_0, ds, rho_tot, rho_tot_0, dsig):
        """m^j(s) = (rho^j/T)(1 + k_turn dsig^2)(1 + k dsig)   (Latorre & Humphrey 2018, Eq. 9, 11)"""
        return (rho / (self.T / ds)) * (1.0 + self.k_turn * dsig ** 2) * (1.0 + self.k * dsig)


@pytree(("law", "M_dep"))
class Reorienting:
    """production of law, each cohort deposited along M_dep of its time (per Gauss point, fcmm.core.sigma_solver)"""
    def __init__(self, law, M_dep):
        self.law, self.M_dep = law, M_dep

    def m(self, *args):
        return self.law.m(*args)
