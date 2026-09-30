"""Production laws: increment(c, sigma_f, ds, rho_tot, rho_tot_0) -> rho_dot_+ ds of constituent c"""

from hcmm.core import pytree, sigma_f_rel


@pytree(("T", "k"))
class MaesProduction:
    def __init__(self, T, k):
        self.T, self.k = T, k

    def increment(self, c, sigma_f, ds, rho_tot, rho_tot_0):
        """rho_dot_+ ds = (rho/T)(1 + k_sigma_+ rel) ds   (Eq. 24)"""
        return (c.rho / (self.T / ds)) * (1.0 + self.k * sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0))
