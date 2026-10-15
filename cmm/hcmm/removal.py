"""Removal laws: increment(c, sigma_f, ds, rho_tot, rho_tot_0, dtau, dsig) -> rho_dot_- ds (<= 0) of constituent c"""

from fem.pytree import pytree
from hcmm.core import sigma_f_rel


@pytree(("T", "k_turn"))
class MixtureRemoval:
    """first-order removal with the stress-dependent turnover rate of Latorre & Humphrey 2018 (Eq. 10, 11)"""
    def __init__(self, T, k_turn=0.0):
        self.T, self.k_turn = T, k_turn

    def increment(self, c, sigma_f, ds, rho_tot, rho_tot_0, dtau, dsig):
        """rho_dot_- ds = -(rho/T)(1 + k_turn dsig^2) ds"""
        return -(c.rho / (self.T / ds)) * (1.0 + self.k_turn * dsig ** 2)


@pytree(("T", "k"))
class MaesRemoval:
    def __init__(self, T, k):
        self.T, self.k = T, k

    def increment(self, c, sigma_f, ds, rho_tot, rho_tot_0, dtau, dsig):
        """rho_dot_- ds = -(rho/T)(1 + k_sigma_- rel) ds   (Eq. 24), independent of dtau, dsig"""
        return -(c.rho / (self.T / ds)) * (1.0 + self.k * sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0))
