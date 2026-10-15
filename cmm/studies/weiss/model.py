"""Weiss et al. 2020 (Sci. Adv. 6, eabd3574), mouse descending thoracic aorta: wall (elastin, smooth muscle, four
collagen families) and parameters of table S2 (Fbln5+/+), insult targets (Fbln5-/-)."""

import numpy as np

from wall import Part, parameters

MMHG = 1.33322e-4
KO = dict(C10=0.0699 / 2, alpha=np.radians(50.3))

WALL_WEISS = (Part("elastin", "neo-hooke", growth="fixed"), Part("smc", "fung", "theta", "turnover"),
              Part("circ", "fung", "theta", "turnover"), Part("axial", "fung", "z", "turnover"),
              Part("diag+", "fung", "+diag", "turnover", reorient=True),
              Part("diag-", "fung", "-diag", "turnover", reorient=True))


def weiss_par(**kw):
    """Table S2 (Fbln5+/+) in MPa, mm, days; "<constituent>.<parameter>" per unit mass fraction:
        W^e = c_e/2 (I1 - 3)                         -> C10 = c_e / 2
        W^k = c1/(4 c2) (exp(c2 (lam^2 - 1)^2) - 1)  -> k1 = c1 / 2, k2 = c2
    elastin (rho = phi_e, C10, deposition stretches G_theta, G_z), smc (rho = phi_m, k1, k2, g = G_m), collagen circ /
    axial / diag+- (rho = phi_c beta, k1, k2, g = G_c, alpha from the axis); turnover of Latorre & Humphrey 2018
    (Weiss ref. 42): T = 1/k_o = 7 days, K_sigma^c = 2, eta = 1 (same for smc), k_tau = 0 (no WSS),
    k_turn = 1: turnover rate k_o (1 + dsig^2) of Latorre & Humphrey 2018 Eq. 11 (0: constant);
    a, h: in vivo radius and thickness, p_hom = p_gr = 120 mmHg"""
    turnover = dict(T=7.0, k_sigma=2.0, k_tau=0.0, k_turn=0.0)
    collagen = lambda beta: dict(rho=0.404 * beta, k1=1.278 / 2, k2=11.1, g=1.09, **turnover)
    diag = dict(collagen(0.893 / 2), alpha=np.radians(38.1))
    groups = {"elastin": dict(rho=0.331, C10=0.0885 / 2, G_theta=2.15, G_z=1.57),
              "smc": dict(rho=0.265, k1=0.4191 / 2, k2=0.105, g=1.20, **turnover),
              "circ": collagen(0.045), "axial": collagen(0.062), "diag+": diag, "diag-": dict(diag)}
    return parameters(groups, dict(a=0.568, h=0.034, p_hom=120 * MMHG, p_gr=120 * MMHG), kw)
