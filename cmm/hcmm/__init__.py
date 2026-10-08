"""Homogenized constrained mixture model (HCMM), Maes & Famaey (2023) sec. 2.2.

core: constituent, mixture, sigma_solver, commit;  growth: F_g laws;  production / removal: turnover laws;
materials: shared library materials/ (passed in via the constituents)
"""

from hcmm.core import constituent, mixture, sigma_solver, commit, J_target, wss
from hcmm.growth import Isotropic, Anisotropic
from hcmm.production import MaesProduction
from hcmm.removal import MaesRemoval


def maes(T, k_plus, k_minus=0.0, k_tau=0.0):
    """(production, removal) of Maes Eq. 24 with the same T, so homeostasis is kept at rel = 0 and tau_w = tau_w_h;
    k_tau: wall shear stress gain of production"""
    return MaesProduction(T, k_plus, k_tau), MaesRemoval(T, k_minus)
