"""HCMM core: constituents, mixture, stress and the commit of one G&R step.

    F_e^j(s)        = F(s) F_g(s)^-1 F_r^j(s)^-1                    (Eq. 15)
    rho^j(s+ds)     = rho^j(s) + rho_dot^j(s) ds                     (Eq. 37)
    (rho_dot_+/rho)(sigma^j - sigma_pre^j)
                    = (d sigma^j / d F_e^j) : (F_e^j L_r^j)          (Eq. 20)

Solver interface: sigma_tot, aux = sigma_solver(state, F); state = commit(state, F, aux)
"""

import jax
import jax.numpy as jnp
from jax import jit
from fem.pytree import pytree
from fem.tensor3 import det3, inv3, polar_rotation


@jit
def F_e_calc(F, F_g, F_r):
    """F_e^j(s) = F(s) F_g(s)^-1 F_r^j(s)^-1   (Eq. 15)"""
    return F @ inv3(F_g) @ inv3(F_r)


@pytree(("material", "production", "removal", "rho", "F_r", "sigma_pre", "sigma_f_pre", "sigma_f", "phi"),
        static=("sigma_pre_mode",))
class constituent:
    def __init__(self, material, rho_0, production=None, removal=None, G=None, sigma_pre=None, F_r=None,
                 sigma_pre_mode="deposition"):
        """production/removal: laws from hcmm.production / hcmm.removal (None: no turnover, e.g. elastin).
        sigma_pre_mode: "deposition" (F_e -> G, Cyron/CMM) or "initial" (Maes Eq. 17)."""
        if sigma_pre_mode not in ("deposition", "initial"):
            raise ValueError(f"sigma_pre_mode must be 'deposition' or 'initial', got {sigma_pre_mode!r}")
        self.sigma_pre_mode = sigma_pre_mode
        self.material = material
        self.production = production
        self.removal = removal
        self.rho = rho_0

        self.sigma_pre = material.sigma(G) if sigma_pre is None else sigma_pre
        self.sigma_f_pre = material.sigma_f(self.sigma_pre)
        self.sigma_f = self.sigma_f_pre

        if F_r is not None:
            self.F_r = F_r
        elif G is not None:
            self.F_r = inv3(G)
        else:
            self.F_r = jnp.eye(3)

        self.phi = jnp.asarray(1.0)

    @property
    def turnover(self):
        """static: None-ness of the laws is part of the pytree structure"""
        return self.production is not None or self.removal is not None


@pytree(("constituents", "ds", "rho_tot_0", "rho_tot", "F_g", "growth"))
class mixture:
    def __init__(self, constituents, ds, growth):
        """growth: law from hcmm.growth, F_g from rho_tot / rho_tot_0"""
        self.constituents = constituents
        self.ds = ds
        self.growth = growth
        self.rho_tot_0 = sum(c.rho for c in constituents)
        self.rho_tot = self.rho_tot_0
        for c in constituents:
            c.phi = c.rho / self.rho_tot_0
        self.F_g = growth.F_g(self.rho_tot / self.rho_tot_0)


def sigma_f_rel(c, sigma_f, rho_tot, rho_tot_0, eps=1e-9):
    """rel = [rho_tot sigma_f(s) - rho_tot(0) sigma_f(0)] / [rho_tot(0) sigma_f(0)]   (Eq. 24)

    Denominator dropped when sigma_f(0) = 0.
    """
    ref = rho_tot_0 * c.sigma_f_pre
    denom = jnp.where(jnp.abs(ref) < eps, 1.0, ref)
    return (rho_tot * sigma_f - ref) / denom


@jit
def sigma_solver(mixt, F):
    """sigma_tot = sum_j phi^j sigma^j = sum_j (rho^j/J) sigma_mat^j(F_e^j)   (Eq. 17, 27)"""
    F_g = mixt.F_g
    J = det3(F)
    sigma_tot = jnp.zeros((3,3))
    trial = []
    for c in mixt.constituents:
        F_e = F_e_calc(F, F_g, c.F_r)
        sigma = c.material.sigma(F_e)
        sigma_tot += sigma * c.rho / J
        trial.append((F_e, sigma))
    return sigma_tot, trial


@jit
def constituent_update(c, F_e, sigma_s, R, ds, J, rho_tot, rho_tot_0):
    """sigma_pre^j(s) = (J_g/J) R sigma^j(0) R^T          deposition (Cyron Eq. 11)
                   = R sigma^j(0) R^T                   initial    (Eq. 17)
    sigma_f^j(s)   = tr(sigma^j)/J                      (Eq. 31)
    rho^j(s+ds)    = rho^j + rho_dot_+ ds + rho_dot_- ds   (Eq. 37)
    F_r^j(s+ds)    from Eq. 20 via Eq. 41 or Eq. 43
    """
    sigma_f_s = c.material.sigma_f(sigma_s) / J
    if not c.turnover:
        return c.rho, c.F_r, sigma_f_s
    sigma_pre_s = R @ c.sigma_pre @ R.T

    zero = jnp.zeros_like(c.rho)
    rho_dot_plus = zero if c.production is None else c.production.increment(c, sigma_f_s, ds, rho_tot, rho_tot_0)
    rho_dot_minus = zero if c.removal is None else c.removal.increment(c, sigma_f_s, ds, rho_tot, rho_tot_0)
    rho_s = c.rho + rho_dot_plus + rho_dot_minus

    rho_pre = rho_tot / J if c.sigma_pre_mode == "deposition" else rho_tot_0
    sigma_rate_euler = (rho_dot_plus / c.rho) * (rho_tot * sigma_s / J
                                                 - rho_pre * sigma_pre_s)
    F_r_s = c.material.F_r(F_e, J, c, sigma_rate_euler)
    return rho_s, F_r_s, sigma_f_s


def J_target(mixt):
    """J = det F_g for incompressible constituents (det F_e = det F_r = 1)"""
    return det3(mixt.F_g)


@jit
def commit(mixt, F, aux):
    """New state: rho^j, F_r^j, phi^j = rho^j/rho_tot, F_g on the settled F."""
    J = det3(F)
    R = polar_rotation(F)
    updates = [constituent_update(c, F_e, sigma_s, R, mixt.ds, J, mixt.rho_tot, mixt.rho_tot_0)
               for c, (F_e, sigma_s) in zip(mixt.constituents, aux)]
    rho_tot_s = sum(u[0] for u in updates)
    constituents = [c.replace(rho=rho_s, F_r=F_r_s, phi=rho_s / rho_tot_s, sigma_f=sigma_f_s)
                    for c, (rho_s, F_r_s, sigma_f_s) in zip(mixt.constituents, updates)]
    return mixt.replace(constituents=constituents, rho_tot=rho_tot_s,
                        F_g=mixt.growth.F_g(rho_tot_s / mixt.rho_tot_0))
