"""External single-element FEM solver for the Maes & Famaey (2023) U/S/F cases.

Model-agnostic. Any model module exposing this pair can be driven by it:

    sigma_tot, aux = model.sigma_solver(state, F)
    state          = model.commit(state, F, aux)

Contract the models must honour:
  * sigma_solver is PURE -- it must not mutate state, and must read only
    committed state. The Newton loop below calls it at many trial F, and all
    but the last of those are thrown away.
  * commit runs exactly ONCE per time step, on the converged F. This is where
    a model does its expensive state update (HCMM: the F_r solve; FCMM: the
    cohort-history append).

This mirrors an Abaqus UMAT split: sigma_solver is what feeds the global
equilibrium iteration, commit is what writes STATEV.

Axis convention (Fig. 1, matching CMM_test_cases.py):
    axis 0   paper Z   S3 & S5 constrained  -> CONFINED, lam = 1
    axis 1   paper Y   S6 fixed, S4 driven  -> DRIVEN (U / S / F applied here)
    axis 2   paper X   S1 fixed, far free   -> FREE, sigma_22 = 0
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass

CONFINED_AXIS, DRIVEN_AXIS, FREE_AXIS = 0, 1, 2


@dataclass
class BC:
    """case 'U': target is a stretch. 'S': a Cauchy stress. 'F': a force.

    A0 is the reference area of the driven face; the paper's 50x50x50 mm
    element gives 2500 mm^2, so its 250 N <-> 0.100 MPa.
    """
    case: str
    target: float
    A0: float = 2500.0

    def __post_init__(self):
        if self.case not in ("U", "S", "F"):
            raise ValueError(f"case must be 'U', 'S' or 'F', got {self.case!r}")

    @property
    def n_unknowns(self):
        """U prescribes the driven stretch, so only the free axis is unknown."""
        return 1 if self.case == "U" else 2


# ==========================================
# Kinematics of the single element
# ==========================================


def F_from_x(x, bc):
    """F = diag(1, lam_driven, lam_free). The confined axis is always 1."""
    if bc.case == "U":
        return jnp.diag(jnp.array([1.0, bc.target, x[0]]))
    return jnp.diag(jnp.array([1.0, x[0], x[1]]))


def x_from_F(F, bc):
    """Inverse of F_from_x, for seeding the next step from the last solution."""
    if bc.case == "U":
        return jnp.array([F[FREE_AXIS, FREE_AXIS]])
    return jnp.array([F[DRIVEN_AXIS, DRIVEN_AXIS], F[FREE_AXIS, FREE_AXIS]])


# ==========================================
# Equilibrium residual
# ==========================================


def residual(x, state, bc, sigma_solver):
    """Traction BCs on the free and driven faces. Uses committed state only."""
    F = F_from_x(x, bc)
    sigma, _ = sigma_solver(state, F)
    free = sigma[FREE_AXIS, FREE_AXIS]

    if bc.case == "U":
        return jnp.array([free])

    if bc.case == "S":
        return jnp.array([sigma[DRIVEN_AXIS, DRIVEN_AXIS] - bc.target, free])

    # Case F is a dead load on the REFERENCE area, so the residual is nominal
    # (1st Piola-Kirchhoff) traction: P_11 = J * sigma_11 / lam_1.
    J = jnp.linalg.det(F)
    P = J * sigma[DRIVEN_AXIS, DRIVEN_AXIS] / F[DRIVEN_AXIS, DRIVEN_AXIS]
    return jnp.array([P * bc.A0 - bc.target, free])


def solve_F(state, bc, sigma_solver, x0, tol=None, max_iter=50):
    """Newton on the free stretches. Returns (F, x, n_iter, converged).

    Not jitted: the loop trip count is data-dependent, and FCMM's history array
    grows every step so a jitted driver would retrace anyway.

    tol defaults to a dtype-aware value -- in float32 a fixed 1e-10 is below
    the round-off floor and Newton just chatters there forever. The step-size
    test is the backstop: once an update is lost in round-off, x is as settled
    as this precision allows.
    """
    x = jnp.asarray(x0, dtype=jnp.result_type(float))
    eps = float(jnp.finfo(x.dtype).eps)

    def r(xx):
        return residual(xx, state, bc, sigma_solver)

    fvec = r(x)
    if tol is None:
        tol = 1e3 * eps * (1.0 + float(jnp.max(jnp.abs(fvec))))

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if float(jnp.max(jnp.abs(fvec))) < tol:
            converged = True
            break
        fjac = jax.jacfwd(r)(x)
        dx = jnp.linalg.solve(fjac, -fvec)
        x = x + dx
        fvec = r(x)
        if float(jnp.max(jnp.abs(dx))) <= eps * (1.0 + float(jnp.max(jnp.abs(x)))):
            converged = True
            break

    return F_from_x(x, bc), x, it, converged


# ==========================================
# Time stepping
# ==========================================


def step(state, bc, model, x0):
    """One time step: settle F against the BC, then commit once on that F."""
    F, x, n_iter, converged = solve_F(state, bc, model.sigma_solver, x0)
    if not converged:
        raise RuntimeError(f"equilibrium Newton did not converge in {n_iter} iterations")

    sigma, aux = model.sigma_solver(state, F)
    state = model.commit(state, F, aux)
    return state, F, x, sigma


def run(state, model, bc, n_steps, x0=None):
    """Drive n_steps of growth and remodeling.

    bc is either a single BC held for every step, or a list of length n_steps
    (the paper's case S/F raise the load at the start of the G&R phase).
    Returns a dict of stacked histories.
    """
    bcs = bc if isinstance(bc, (list, tuple)) else [bc] * n_steps
    if len(bcs) != n_steps:
        raise ValueError(f"got {len(bcs)} BCs for {n_steps} steps")

    if x0 is None:
        x0 = jnp.ones(bcs[0].n_unknowns)

    F_hist, sigma_hist, x_hist = [], [], []
    for k in range(n_steps):
        state, F, x, sigma = step(state, bcs[k], model, x0)
        x0 = x  # warm-start the next step from this solution
        F_hist.append(F)
        sigma_hist.append(sigma)
        x_hist.append(x)

    return {
        "state": state,
        "F": jnp.stack(F_hist),
        "sigma": jnp.stack(sigma_hist),
        "x": jnp.stack(x_hist),
        "lam_driven": jnp.stack([F[DRIVEN_AXIS, DRIVEN_AXIS] for F in F_hist]),
        "sigma_driven": jnp.stack([s[DRIVEN_AXIS, DRIVEN_AXIS] for s in sigma_hist]),
    }
