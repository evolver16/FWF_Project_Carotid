"""Production laws: increment(c, sigma_f, ds, rho_tot, rho_tot_0, dtau) -> rho_dot_+ ds of constituent c,
dtau = tau_w / tau_w_h - 1 (wall shear stress stimulus, 0 without flow input)"""

from fem.pytree import pytree
from hcmm.core import sigma_f_rel


@pytree(("T", "k", "k_tau"))
class MaesProduction:
    def __init__(self, T, k, k_tau=0.0):
        self.T, self.k, self.k_tau = T, k, k_tau

    def increment(self, c, sigma_f, ds, rho_tot, rho_tot_0, dtau):
        """rho_dot_+ ds = (rho/T)(1 + k_sigma_+ rel - k_tau dtau) ds   (Eq. 24; WSS term as Latorre & Humphrey 2018)"""
        return (c.rho / (self.T / ds)) * (1.0 + self.k * sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0) - self.k_tau * dtau)
