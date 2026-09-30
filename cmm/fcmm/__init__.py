"""Full constrained mixture model (FCMM), Maes & Famaey (2023) sec. 2.1.

core: cohort histories, constituent, mixture, commit;  growth: F_g laws;  production / removal: turnover laws;
materials: shared library materials/ (passed in via params)
"""

from fcmm.core import params, constituent, mixture, history, mixture_history, sigma_solver, commit, window_for
from fcmm.growth import Isotropic, Anisotropic
from fcmm.production import MaesProduction
from fcmm.removal import MaesRemoval


def maes(T, k_plus, k_minus=0.0):
    """(production, removal) of Maes Eq. 11/13 with the same T, so homeostasis is kept at rel = 0"""
    return MaesProduction(T, k_plus), MaesRemoval(T, k_minus)
