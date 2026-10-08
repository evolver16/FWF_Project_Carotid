"""Removal laws: increment(c, sigma_f, ds, rho_tot, rho_tot_0, dtau) -> rho_dot_- ds (<= 0) of constituent c"""

from fem.pytree import pytree
from hcmm.core import sigma_f_rel


@pytree(("T", "k"))
class MaesRemoval:
    def __init__(self, T, k):
        self.T, self.k = T, k

    def increment(self, c, sigma_f, ds, rho_tot, rho_tot_0, dtau):
        """rho_dot_- ds = -(rho/T)(1 + k_sigma_- rel) ds   (Eq. 24), independent of dtau"""
        return -(c.rho / (self.T / ds)) * (1.0 + self.k * sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0))
