"""HCMM core, fully implicit growth and remodeling (Sempertegui & Avril 2023).

Unknowns per constituent with turnover: z^j (remodeling increment) and rho^j (density at n+1),
all solved in one Newton, coupled through F_g(rho_tot)   (Eq. 37)

    F_g          = F_g(sum_j rho^j / rho_tot(0))                                          (Eq. 4)
    sigma_trial  = sigma_mat(F F_g^-1 F_r(n)^-1)                                          (old F_r, new F, F_g)
    sigma_target = [drho_+ sigma_pre + rho sigma_trial] / (drho_+ + rho)                   (Eq. 18)
    F_r(n+1)     = exp(sum_i z_i B_i) F_r(n),  B_i sym., tr B_i = 0                         (Eq. 28)
    remodeling:  dev(sigma(F_r(n+1)) - sigma_target) = 0                                  (Eq. 20)
    growth:      rho(n+1) - rho(n) - drho_+ - drho_- = 0,  increments at n+1              (Eq. 33)
    sigma_tot    = sum_j (rho^j/J) sigma_mat^j(F_e^j)                                     (Eq. 17, 27)

Solver interface: sigma_tot, aux = sigma_solver(state, F); state = commit(state, F, aux)
"""

from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree
from fem.pytree import pytree
from fem.tensor3 import det3, inv3, polar_rotation

_a, _b = np.sqrt(0.5), np.sqrt(1.0 / 6.0)
DEV_BASIS = np.array([
    [[_a, 0, 0], [0, -_a, 0], [0, 0, 0]],
    [[_b, 0, 0], [0, _b, 0], [0, 0, -2 * _b]],
    [[0, _a, 0], [_a, 0, 0], [0, 0, 0]],
    [[0, 0, _a], [0, 0, 0], [_a, 0, 0]],
    [[0, 0, 0], [0, 0, _a], [0, _a, 0]],
])
SLACK_EPS = 1e-12


@jit
def F_e_calc(F, F_g, F_r):
    """F_e^j(s) = F(s) F_g(s)^-1 F_r^j(s)^-1   (Eq. 15)"""
    return F @ inv3(F_g) @ inv3(F_r)


@pytree(("material", "production", "removal", "rho", "F_r", "C_r", "sigma_pre", "sigma_f_pre", "sigma_f", "phi"),
        static=("sigma_pre_mode",))
class constituent:
    def __init__(self, material, rho_0, production=None, removal=None, G=None, sigma_pre=None, F_r=None, C_r=None,
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
        elif C_r is not None:
            w, V = jnp.linalg.eigh(C_r)
            self.F_r = V @ jnp.diag(jnp.sqrt(w)) @ V.T
        elif G is not None:
            self.F_r = inv3(G)
        else:
            self.F_r = jnp.eye(3)
        self.C_r = self.F_r.T @ self.F_r

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


def flow_basis(c):
    """directions B_i in which F_r may change: 5 deviatoric, or the material's own (Fung: dev(M(x)M))"""
    return getattr(c.material, "flow_basis", DEV_BASIS)


def F_r_new(c, z):
    """F_r(n+1) = exp(Z) F_r(n),  Z = z_i B_i   (Eq. 28)
    closed form if the material provides flow_exp, else exp(Z) = [sum_k=0..12 (Z/16)^k / k!]^16
    """
    if hasattr(c.material, "flow_exp"):
        return c.material.flow_exp(z) @ c.F_r
    A = jnp.tensordot(z, flow_basis(c), 1) / 16.0
    exp_Z = jnp.eye(3)
    term = jnp.eye(3)
    for k in range(1, 13):
        term = term @ A / k
        exp_Z = exp_Z + term
    for _ in range(4):
        exp_Z = exp_Z @ exp_Z
    return exp_Z @ c.F_r


def mismatch(c, F, F_g, sigma_ref, z):
    """[F_e^T (sigma(z) - sigma_ref) F_e^-T] : B_i   (Eq. 20, 24)"""
    F_e = F_e_calc(F, F_g, F_r_new(c, z))
    mandel = F_e.T @ (c.material.sigma(F_e) - sigma_ref) @ inv3(F_e).T
    return jnp.tensordot(flow_basis(c), mandel, 2)


ConstituentState = namedtuple("ConstituentState", "rho F_r sigma sigma_f residual")
Step = namedtuple("Step", "F F_g R J rho_tot rho_tot_0 ds")


@jit
def sigma_solver(mixt, F):
    """Newton on x = (z^j, rho^j) of all constituents, then sigma_tot = sum_j (rho^j/J) sigma^j   (Eq. 17, 27)
    x = x* - J(x*)^-1 r(x*, F),  x*, J frozen:  dx/dF = -J^-1 dr/dF   (implicit function theorem)
    """
    R = polar_rotation(F)
    J = det3(F)

    guess, slack = [], []
    for c in mixt.constituents:
        if c.turnover:
            z = jnp.zeros(flow_basis(c).shape[0])
            sigma_trial = c.material.sigma(F_e_calc(F, mixt.F_g, c.F_r))
            stiffness = jnp.diag(jax.jacfwd(mismatch, argnums=4)(c, F, mixt.F_g, sigma_trial, z))
            guess.append((z, jnp.asarray(c.rho, float)))
            slack.append(jnp.abs(stiffness) < SLACK_EPS)
        else:
            guess.append(None)
            slack.append(None)

    flat_guess, unflatten = ravel_pytree(guess)

    rho_columns, blocks, offset = [], [], 0
    for u in guess:
        if u is not None:
            n = u[0].shape[0]
            blocks.append(np.arange(offset, offset + n + 1))
            rho_columns.append(offset + n)
            offset += n + 1
    rho_columns = np.array(rho_columns, dtype=int)

    def residual(x, mixt, F, R, J):
        states = constituent_states(mixt, F, R, J, unflatten(x), slack)
        return ravel_pytree([s.residual for s in states])[0]

    def jacobian(x, mixt, F, R, J):
        """z^a only enters the residual of a, rho^b (b != a) only through rho_tot (F_g, turnover law):
        per constituent d r^a / d(z^a, rho^a, rho_others),  d r^a / d rho^b = d r^a / d rho_others
        """
        unknowns = unflatten(x)
        rho_tot = growth_step(mixt, F, R, J, unknowns).rho_tot
        coupled = len(blocks) > 1
        jac = jnp.zeros((x.size, x.size))
        rows = iter(blocks)
        for c, u, s in zip(mixt.constituents, unknowns, slack):
            if u is None:
                continue
            n = u[0].shape[0]
            rho_others = rho_tot - u[1]

            def own_residual(v):
                rho_tot_v = v[n] + (v[n + 1] if coupled else rho_others)
                step = Step(F=F, F_g=mixt.growth.F_g(rho_tot_v / mixt.rho_tot_0), R=R, J=J,
                            rho_tot=rho_tot_v, rho_tot_0=mixt.rho_tot_0, ds=mixt.ds)
                return constituent_state(c, (v[:n], v[n]), step, s).residual

            v = jnp.concatenate([u[0], u[1][None]] + ([rho_others[None]] if coupled else []))
            d = jax.jacfwd(own_residual)(v)
            block = next(rows)
            jac = jac.at[np.ix_(block, block)].set(d[:, :n + 1])
            if coupled:
                others = np.setdiff1d(rho_columns, block[-1])
                jac = jac.at[np.ix_(block, others)].set(jnp.broadcast_to(d[:, n + 1:], (n + 1, others.size)))
        return jac

    solution = guess
    if flat_guess.size:
        frozen = jax.lax.stop_gradient((mixt, F, R, J))
        x = newton(residual, jacobian, jax.lax.stop_gradient(flat_guess), frozen)
        x = x - jnp.linalg.solve(jacobian(x, *frozen), residual(x, mixt, F, R, J))
        solution = unflatten(x)

    states = constituent_states(mixt, F, R, J, solution, slack)

    sigma_tot = sum(s.rho * s.sigma for s in states) / J
    return sigma_tot, [(s.rho, s.F_r, s.sigma_f) for s in states]


def growth_step(mixt, F, R, J, unknowns):
    """rho_tot and F_g from the guessed densities   (Eq. 4)"""
    densities = []
    for c, u in zip(mixt.constituents, unknowns):
        densities.append(c.rho if u is None else u[1])
    rho_tot = sum(densities)
    F_g = mixt.growth.F_g(rho_tot / mixt.rho_tot_0)
    return Step(F=F, F_g=F_g, R=R, J=J, rho_tot=rho_tot, rho_tot_0=mixt.rho_tot_0, ds=mixt.ds)


def constituent_states(mixt, F, R, J, unknowns, slack):
    """growth F_g from the guessed densities, then constituent_state of every constituent"""
    step = growth_step(mixt, F, R, J, unknowns)
    states = []
    for c, u, s in zip(mixt.constituents, unknowns, slack):
        states.append(constituent_state(c, u, step, s))
    return states


def constituent_state(c, unknowns, step, slack):
    """state and residual of constituent c for its unknowns (z, rho) in the current step
    sigma_pre       = R sigma_pre R^T  deposition,  (J rho_tot(0)/rho_tot) R sigma_pre R^T  initial
    sigma_target    = [drho_+ sigma_pre + rho sigma_trial] / (drho_+ + rho)              (Eq. 18)
    remodeling_i    = [F_e^T (sigma(z) - sigma_target) F_e^-T] : B_i,  z_i = 0 if slack   (Eq. 20, 24)
    growth          = rho - rho(n) - drho_+ - drho_-                                    (Eq. 33)
    """
    F, F_g = step.F, step.F_g

    sigma_trial = c.material.sigma(F_e_calc(F, F_g, c.F_r))
    if not c.turnover:
        return ConstituentState(c.rho, c.F_r, sigma_trial, c.material.sigma_f(sigma_trial) / step.J, jnp.zeros(0))

    z, rho = unknowns
    F_r = F_r_new(c, z)
    sigma = c.material.sigma(F_e_calc(F, F_g, F_r))
    sigma_f = c.material.sigma_f(sigma) / step.J

    c_new = c.replace(rho=rho)
    args = (sigma_f, step.ds, step.rho_tot, step.rho_tot_0)
    zero = jnp.zeros_like(rho)
    drho_plus = zero if c.production is None else jnp.maximum(c.production.increment(c_new, *args), 0.0)
    drho_minus = zero if c.removal is None else c.removal.increment(c_new, *args)

    scale = 1.0 if c.sigma_pre_mode == "deposition" else step.J * step.rho_tot_0 / step.rho_tot
    sigma_pre = scale * step.R @ c.sigma_pre @ step.R.T
    sigma_target = (drho_plus * sigma_pre + rho * sigma_trial) / (drho_plus + rho)

    remodeling = jnp.where(slack, z, mismatch(c, F, F_g, sigma_target, z))
    growth = rho - c.rho - drho_plus - drho_minus
    return ConstituentState(rho, F_r, sigma, sigma_f, jnp.append(remodeling, growth))


def newton(residual, jacobian, x, args, tol=1e-8, max_iter=25):
    """x <- x - J(x)^-1 r(x) until max |r| < tol (1 + max |r(x_0)|)"""
    def step(state):
        x, r, it = state
        x = x - jnp.linalg.solve(jacobian(x, *args), r)
        return x, residual(x, *args), it + 1

    r = residual(x, *args)
    tol = tol * (1.0 + jnp.max(jnp.abs(r)))

    def not_converged(state):
        _, r, it = state
        return (jnp.max(jnp.abs(r)) > tol) & (it < max_iter)

    return jax.lax.while_loop(not_converged, step, (x, r, 0))[0]


def J_target(mixt):
    """J = det F_g for incompressible constituents (det F_e = det F_r = 1)"""
    return det3(mixt.F_g)


@jit
def commit(mixt, F, aux):
    """stores what sigma_solver computed at the converged F: rho^j, F_r^j, C_r^j = F_r^T F_r, phi^j, sigma_f^j, F_g"""
    rho_tot = sum(rho for rho, _, _ in aux)
    constituents = [c.replace(rho=rho, F_r=F_r, C_r=F_r.T @ F_r, phi=rho / rho_tot, sigma_f=sigma_f)
                    for c, (rho, F_r, sigma_f) in zip(mixt.constituents, aux)]
    return mixt.replace(constituents=constituents, rho_tot=rho_tot,
                        F_g=mixt.growth.F_g(rho_tot / mixt.rho_tot_0))
