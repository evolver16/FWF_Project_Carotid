"""Production laws: increment(c, sigma_f, ds, rho_tot, rho_tot_0, dtau, dsig) -> rho_dot_+ ds of constituent c,
dtau = tau_w / tau_w_h - 1 (wall shear stress stimulus, 0 without flow input),
dsig = sigma_w / sigma_w_h - 1 (wall stress stimulus, 0 without input)"""

from fem.pytree import pytree
from hcmm.core import sigma_f_rel


@pytree(("T", "k", "k_tau"))
class MaesProduction:
    def __init__(self, T, k, k_tau=0.0):
        self.T, self.k, self.k_tau = T, k, k_tau

    def increment(self, c, sigma_f, ds, rho_tot, rho_tot_0, dtau, dsig):
        """rho_dot_+ ds = (rho/T)(1 + k_sigma_+ rel - k_tau dtau) ds   (Eq. 24; WSS term as Latorre & Humphrey 2018)"""
        return (c.rho / (self.T / ds)) * (1.0 + self.k * sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0) - self.k_tau * dtau)


@pytree(("T", "k", "k_tau"))
class MixtureProduction:
    """stimulus from the wall stress (Latorre & Humphrey 2018): set by equilibrium (~ P r / h), so restored only by mass,
    not by remodeling of the constituents as the fiber stress in MaesProduction"""
    def __init__(self, T, k, k_tau=0.0):
        self.T, self.k, self.k_tau = T, k, k_tau

    def increment(self, c, sigma_f, ds, rho_tot, rho_tot_0, dtau, dsig):
        """rho_dot_+ ds = (rho/T)(1 + k dsig - k_tau dtau) ds"""
        return (c.rho / (self.T / ds)) * (1.0 + self.k * dsig - self.k_tau * dtau)
