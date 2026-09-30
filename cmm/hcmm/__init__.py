"""Homogenized constrained mixture model (HCMM), Maes & Famaey (2023) sec. 2.2.

core: constituent, mixture, commit;  growth: F_g laws;  production / removal: turnover laws;  materials: shared library materials/
"""

from hcmm.core import constituent, mixture, sigma_solver, commit, J_target, F_e_calc, polar_rotation
from hcmm.growth import Isotropic, Anisotropic
from materials import Fung, NeoHookean, NeoHookeanInc
from hcmm.production import MaesProduction
from hcmm.removal import MaesRemoval


def maes(T, k_plus, k_minus=0.0):
    """(production, removal) of Maes Eq. 24 with the same T, so homeostasis is kept at rel = 0"""
    return MaesProduction(T, k_plus), MaesRemoval(T, k_minus)
