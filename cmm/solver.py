"""Single-element solver for the Maes & Famaey (2023) U/S/F cases (Fig. 1).

Model interface:
    sigma_tot, aux = model.sigma_solver(state, F)   pure, committed state only
    state          = model.commit(state, F, aux)    once per step, converged F

Axes: 0 = paper Z (confined, lam = 1), 1 = paper Y (driven), 2 = paper X (free).
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, replace
from tensor3 import det3

CONFINED_AXIS, DRIVEN_AXIS, FREE_AXIS = 0, 1, 2


@dataclass
class BC:
    """U: stretch, S: Cauchy stress, F: force on reference area A0 (50x50 mm -> 2500 mm^2)."""
    case: str
    target: float
    A0: float = 2500.0
    hybrid: bool = False

    def __post_init__(self):
        if self.case not in ("U", "S", "F"):
            raise ValueError(f"case must be 'U', 'S' or 'F', got {self.case!r}")

    @property
    def n_unknowns(self):
        return (1 if self.case == "U" else 2) - (1 if self.hybrid else 0)


def F_from_x(x, bc):
    """F = diag(1, lam_driven, lam_free); hybrid: lam_free = 1/lam_driven"""
    lam = bc.target if bc.case == "U" else x[0]
    if bc.hybrid:
        return jnp.diag(jnp.array([1.0, lam, 1.0 / lam]))
    return jnp.diag(jnp.array([1.0, lam, x[-1]]))


def total_sigma(state, F, bc, sigma_solver):
    """sigma - p I, p = sigma_free when hybrid   (Eq. 33 note)"""
    sigma, aux = sigma_solver(state, F)
    if bc.hybrid:
        sigma = sigma - sigma[FREE_AXIS, FREE_AXIS] * jnp.eye(3)
    return sigma, aux


def residual(x, state, bc, sigma_solver):
    """Traction residuals; case F uses nominal traction P = J sigma / lam."""
    F = F_from_x(x, bc)
    sigma, _ = total_sigma(state, F, bc, sigma_solver)
    free = [] if bc.hybrid else [sigma[FREE_AXIS, FREE_AXIS]]

    if bc.case == "U":
        return jnp.array(free)
    if bc.case == "S":
        return jnp.array([sigma[DRIVEN_AXIS, DRIVEN_AXIS] - bc.target] + free)
    J = det3(F)
    P = J * sigma[DRIVEN_AXIS, DRIVEN_AXIS] / F[DRIVEN_AXIS, DRIVEN_AXIS]
    return jnp.array([P * bc.A0 - bc.target] + free)


_RES_JAC = {}


def residual_and_jacobian(bc, sigma_solver):
    """Jitted (x, state, target) -> (r, dr/dx), compiled once per BC type and model."""
    key = (bc.case, bc.hybrid, bc.A0, sigma_solver)
    if key not in _RES_JAC:
        def r(x, state, target):
            return residual(x, state, replace(bc, target=target), sigma_solver)

        _RES_JAC[key] = jax.jit(lambda x, state, target: (r(x, state, target),
                                                           jax.jacfwd(r)(x, state, target)))
    return _RES_JAC[key]


def solve_F(state, bc, sigma_solver, x0, tol=None, max_iter=50):
    """Newton on the free stretches -> (F, x, n_iter, converged)."""
    x = jnp.asarray(x0, dtype=jnp.result_type(float))
    if x.shape[0] == 0:
        return F_from_x(x, bc), x, 0, True
    eps = float(jnp.finfo(x.dtype).eps)
    res_jac = residual_and_jacobian(bc, sigma_solver)
    target = jnp.asarray(bc.target, dtype=x.dtype)

    fvec, fjac = res_jac(x, state, target)
    if tol is None:
        tol = 1e3 * eps * (1.0 + float(jnp.max(jnp.abs(fvec))))

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if float(jnp.max(jnp.abs(fvec))) < tol:
            converged = True
            break
        dx = jnp.linalg.solve(fjac, -fvec)
        x = x + dx
        fvec, fjac = res_jac(x, state, target)
        if float(jnp.max(jnp.abs(dx))) <= eps * (1.0 + float(jnp.max(jnp.abs(x)))):
            converged = True
            break

    return F_from_x(x, bc), x, it, converged


def step(state, bc, model, x0):
    """Equilibrate F, then commit once."""
    F, x, n_iter, converged = solve_F(state, bc, model.sigma_solver, x0)
    if not converged:
        raise RuntimeError(f"equilibrium Newton did not converge in {n_iter} iterations")
    sigma, aux = total_sigma(state, F, bc, model.sigma_solver)
    return model.commit(state, F, aux), F, x, sigma


def run(state, model, bc, n_steps, x0=None):
    """n_steps of G&R under one BC (or a list of BCs) -> stacked histories."""
    bcs = bc if isinstance(bc, (list, tuple)) else [bc] * n_steps
    if len(bcs) != n_steps:
        raise ValueError(f"got {len(bcs)} BCs for {n_steps} steps")
    if x0 is None:
        x0 = jnp.ones(bcs[0].n_unknowns)

    F_hist, sigma_hist = [], []
    for k in range(n_steps):
        state, F, x0, sigma = step(state, bcs[k], model, x0)
        F_hist.append(F)
        sigma_hist.append(sigma)

    F_hist, sigma_hist = jnp.stack(F_hist), jnp.stack(sigma_hist)
    return {
        "state": state,
        "F": F_hist,
        "sigma": sigma_hist,
        "lam_driven": F_hist[:, DRIVEN_AXIS, DRIVEN_AXIS],
        "lam_free": F_hist[:, FREE_AXIS, FREE_AXIS],
        "sigma_driven": sigma_hist[:, DRIVEN_AXIS, DRIVEN_AXIS],
    }
