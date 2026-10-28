"""Model E (Maes & Famaey 2023) on an artery, shared by the artery, fe_gr and vessel studies: parameters, wall,
quarter-cylinder mesh, prestress + G&R run."""

import jax.numpy as jnp
import numpy as np

import fem
import hcmm
import setups
from fem import mesh as meshlib
from setups import ALPHA, FIBER, MATRIX_DE
from studies.common import surface_area
from verification.fem_tube import cylinder_bc, radial_u
from wall import Part, assemble, parameters


def artery_par(**kw):
    """Model E on the artery, one parameter set per constituent, "<constituent>.<parameter>" (per unit mass fraction):
    elastin (rho, C10, K), collagen circ / axial / diag+ / diag- (rho, k1, k2, g, diagonals: alpha from the axis;
    turnover: T in days, k_plus, k_minus, k_tau (WSS gain), k_sigma (wall stress gain)); loads g_ax, p_hom, p_gr.
    kw: a full name sets one value, a bare parameter name (e.g. k_tau) sets it in every constituent that has it"""
    collagen = dict(rho=FIBER["rho_0"], k1=FIBER["k1"], k2=FIBER["k2"], g=FIBER["g"], T=setups.T_DAYS, k_plus=0.1,
                    k_minus=0.0, k_tau=0.0, k_sigma=1.0)
    groups = {"elastin": dict(rho=MATRIX_DE["rho_0"], C10=MATRIX_DE["C10"], K=MATRIX_DE["K"]), "circ": collagen,
              "axial": collagen, "diag+": dict(collagen, alpha=0.5 * jnp.pi - ALPHA),
              "diag-": dict(collagen, alpha=0.5 * jnp.pi - ALPHA)}
    return parameters(groups, dict(g_ax=1.2, p_hom=0.010, p_gr=0.015), kw)


WALL_E = (Part("elastin", "neo-hooke"), Part("circ", "fung", "theta", "turnover"), Part("axial", "fung", "z", "turnover"),
          Part("diag+", "fung", "+diag", "turnover"), Part("diag-", "fung", "-diag", "turnover"))


def artery_mesh(n=(8, 60, 4), element="standard"):
    """(mesh, System, local bases (e_r, e_theta, e_z) at the Gauss points)"""
    m = meshlib.quarter_cylinder(n=n)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces[(0, -1)], element=element)
    return m, sysm, jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))


def wall_area(m, u):
    """4 x area of the deformed end face z = 0"""
    return 4 * surface_area(m, u, m.faces[(2, -1)])


def artery_simulate(m, sysm, Qloc, par, mode="deposition", n_steps=100, bc=None, measure=None,
                    pre_tol=1e-6, pre_iter=60, anderson=5, tol=1e-10, log=None, final=False,
                    checkpoint=None, record=None, bc_gr=None, insult=None, ds=10.0, wall=WALL_E, lesion=None,
                    laws="fiber", incompressible=False, growth="radial", G_e=None, fields=None, n_max=0):
    """Prestress G_elas <- F G_elas at p_hom until u = 0 (Maes UMAT_DEP; Anderson-accelerated, 0 = plain), from G_e
    (per Gauss point; default diag(g_ax^-1/2, g_ax^-1/2, g_ax) in the local basis),
    G&R at p_gr, steps of ds days -> measure(u) per step (default lambda_theta(inner), wall area),
    reverse-differentiable in par; final: also (u, states) at the end; record(k, u, states): after prestress (k = 0) and step k (forward only);
    bc_gr: G&R BCs per step instead of bc(p_gr); insult(k, states) -> states: after the prestress (k = 0) and after
    each step k (e.g. a ramped material change; checkpoint unused);
    wall: constituents, lesion: lesion fraction per Gauss point, laws / incompressible / growth / fields: wall defaults
    and per-Gauss-point parameters (wall.assemble), in the prestress as well; n_max > 0: FCMM (sysm with fcmm)"""
    if bc is None:
        bc = lambda p: cylinder_bc(m, p)
    if measure is None:
        a = float(np.linalg.norm(m.X[m.nodes[(0, -1)][0], :2]))
        measure = lambda u: (1 + radial_u(m, u, -1) / a, wall_area(m, u))
    if G_e is None:
        g_ax = par["g_ax"]
        G_e = Qloc @ jnp.diag(jnp.stack([g_ax ** -0.5, g_ax ** -0.5, g_ax])) @ jnp.swapaxes(Qloc, -1, -2)
    zero = jnp.zeros(sysm.n_dof)
    xs, fs = [], []
    for it in range(1, pre_iter + 1):
        states = assemble(G_e, Qloc, par, wall, mode, ds=ds, lesion=lesion, laws=laws, incompressible=incompressible,
                          growth=growth, fields=fields, n_max=n_max)
        u = sysm.equilibrium(states, bc(par["p_hom"]), zero, tol)
        du = float(np.abs(sysm.last_u).max())
        if log and (it % 5 == 0 or du < pre_tol):
            log(f"  prestress iteration {it:2d}  max|u| {du:.2e} mm  (Newton {sysm.last_iterations})")
        if du < pre_tol:
            break
        xs, fs = (xs + [G_e.ravel()])[-anderson - 1:], (fs + [(sysm.gauss_F(u) @ G_e - G_e).ravel()])[-anderson - 1:]
        G_e = fem.anderson(xs, fs).reshape(G_e.shape)

    if insult:
        states = insult(0, states)
    on_step = lambda u, _: measure(u)
    if record:
        record(0, u, states)
        k = iter(range(1, n_steps + 1))
        on_step = lambda u, st: (record(next(k), u, st), measure(u))[1]
    bcs = bc_gr or [bc(par["p_gr"])] * n_steps
    if insult:
        u, hist = zero, []
        for j, b in enumerate(bcs, 1):
            u, states, h = sysm.run(states, [b], u0=u, tol=tol, on_step=on_step)
            states, hist = insult(j, states), hist + h
    else:
        u, states, hist = sysm.run(states, bcs, u0=zero, tol=tol, on_step=on_step, checkpoint=checkpoint)
    series = tuple(jnp.stack(v) for v in zip(*hist))
    return (series, u, states) if final else series
