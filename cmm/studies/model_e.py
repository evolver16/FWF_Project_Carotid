"""Model E (Maes & Famaey 2023) on an artery, shared by the artery, fe_gr and vessel studies: parameters, wall,
quarter-cylinder mesh, prestress + G&R run."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fem
import hcmm
import setups
from fem import mesh as meshlib
from setups import ALPHA, FIBER, MATRIX_DE
from studies.common import surface_area
from verification.fem_tube import cylinder_bc, radial_u
from wall import Part, assemble, parameters, principal_frame


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
                    laws="fiber", incompressible=False, growth="radial", G_e=None, fields=None, n_max=0, adapt=False, deposit="elastin", beta0=0.0,
                    momentum=0.0, pre_record=None, pre_cache=None):
    """Prestress G_elas <- F G_elas at p_hom until u = 0 (Maes UMAT_DEP; Anderson-accelerated, 0 = plain), from G_e
    (per Gauss point; default diag(g_ax^-1/2, g_ax^-1/2, g_ax) in the local basis);
    adapt (HCMM): also homeostatic fibre content in the same fixed point, the geometry stays the reference:
        rho^j <- rho^j sigma_f^j / sigma_f^j_h          (every fung part, per Gauss point, log step within +-0.3,
                                                       factor in [0.05, 20])
        diag+- at alpha +- beta about the in-plane minor principal direction (wall.principal_frame, deformed basis);
    deposit "elastin": beta once from the equilibrium at p_hom of a stiff stress-free elastin (G = I, C10 x 100, fibre
    masses x 1e-3: u -> 0, the stress balancing p_hom on the reference geometry), then fixed;
    "iterate": beta in the fixed point from beta0 (per Gauss point),
        beta step x min(1, (sigma_c - sigma_a) / (0.1 (|sigma_c| + |sigma_a|)))   (no axis where in-plane isotropic);
    until u = 0 (and beta settled, 1e-3 rad);
    momentum mu: x+ = anderson(x, f) + mu (x - x_prev); pre_record(it, u, states, gauss fields mass_<part>, beta_deg):
    prestress iterations 1, 5, 10, ... and the last; pre_cache: .npz of the converged (G_e, fibre masses, beta), loaded
    as the start if it exists (no elastin solve), written otherwise;
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
    fib = [j for j, p in enumerate(wall) if p.material == "fung"] if adapt else []
    sgn = {j: 1.0 if wall[j].direction == "+diag" else -1.0 for j in fib if wall[j].reorient}
    shp, n_G = G_e.shape[:2], G_e.size
    log_m, beta = jnp.zeros((len(fib),) + shp), jnp.broadcast_to(jnp.asarray(beta0, dtype=float), shp)
    fl, step = fields, 0.0

    def frame(u, states):
        F, sig = sysm.gauss_F(u), sysm.stress(u, states)
        e = [F @ Qloc[..., :, i][..., None] for i in (1, 2)]
        e = [v[..., 0] / jnp.linalg.norm(v[..., 0], axis=-1, keepdims=True) for v in e]
        s = lambda a, b: jnp.einsum("...i,...ij,...j->...", a, sig, b)
        return principal_frame(s(e[0], e[0]), s(e[1], e[1]), s(e[0], e[1]))

    cached = pre_cache is not None and Path(pre_cache).exists()
    if cached:
        z = np.load(pre_cache)
        G_e, log_m, beta = (jnp.asarray(z[k]) for k in ("G_e", "log_m", "beta"))
        if log:
            log(f"  prestress from {Path(pre_cache).name}")
    if adapt and deposit == "elastin" and not cached:
        st0 = assemble(jnp.broadcast_to(jnp.eye(3), G_e.shape), Qloc, par, wall, mode, ds=ds, lesion=lesion,
                       laws=laws, incompressible=incompressible, growth=growth, n_max=n_max, fields=dict(fields or {}) | {
                           f"{wall[j].name}.rho": 1e-3 * par[f"{wall[j].name}.rho"] for j in fib} | {
                           f"{p.name}.C10": 100 * (fields or {}).get(f"{p.name}.C10", par[f"{p.name}.C10"])
                           for p in wall if p.material == "neo-hooke"})
        u0 = sysm.equilibrium(st0, bc(par["p_hom"]), zero, tol)
        beta = frame(u0, st0)[0]
        if log:
            log(f"  fibre directions from the elastin solve: max|u| {float(jnp.abs(u0).max()):.2e} mm, max|beta| "
                f"{np.degrees(float(jnp.abs(beta).max())):.1f} deg")
    for it in range(1, pre_iter + 1):
        if adapt:
            fl = dict(fields or {}) | {f"{wall[j].name}.rho": par[f"{wall[j].name}.rho"] * jnp.exp(log_m[i])
                                       for i, j in enumerate(fib)} \
                 | {f"{wall[j].name}.alpha": par[f"{wall[j].name}.alpha"] + s * beta for j, s in sgn.items()}
        states = assemble(G_e, Qloc, par, wall, mode, ds=ds, lesion=lesion, laws=laws, incompressible=incompressible,
                          growth=growth, fields=fl, n_max=n_max)
        u = sysm.equilibrium(states, bc(par["p_hom"]), zero, tol)
        du = float(np.abs(sysm.last_u).max())
        F = sysm.gauss_F(u)
        x, f = [G_e.ravel()], [(F @ G_e - G_e).ravel()]
        db = 0.0
        if adapt:
            rel = jax.vmap(jax.vmap(lambda st, F: (lambda tr: jnp.stack(
                [st.constituents[j].material.sigma_f(tr[j][1]) / st.constituents[j].sigma_f_pre for j in fib]))(
                hcmm.sigma_solver(st, F)[1])))(states, F)
            x += [log_m.ravel()]
            f += [jnp.clip(jnp.log(jnp.maximum(jnp.moveaxis(rel, -1, 0), 1e-6)), -0.3, 0.3).ravel()]
            if deposit == "iterate":
                b_new, s_c, s_a = frame(u, states)
                w = jnp.clip((s_c - s_a) / (0.1 * (jnp.abs(s_c) + jnp.abs(s_a)) + 1e-12), 0.0, 1.0)
                d_b = w * ((b_new - beta + 0.5 * jnp.pi) % jnp.pi - 0.5 * jnp.pi)
                db = float(jnp.abs(d_b).max())
                x += [beta.ravel()]
                f += [d_b.ravel()]
        if log and (it % 5 == 0 or du < pre_tol):
            un = np.linalg.norm(np.asarray(sysm.last_u).reshape(-1, 3), axis=1)
            log(f"  prestress iteration {it:2d}  max|u| {du:.2e} mm, nodal |u| mean {un.mean():.2e} / 99% "
                f"{np.quantile(un, 0.99):.2e} mm  (Newton {sysm.last_iterations})"
                + (f"  mass factor {float(jnp.exp(log_m).min()):.2f}-{float(jnp.exp(log_m).max()):.2f}, max|beta| "
                   f"{np.degrees(float(jnp.abs(beta).max())):.1f} deg (change {np.degrees(db):.2f})" if adapt else ""))
        if pre_record and (it == 1 or it % 5 == 0 or du < pre_tol):
            pre_record(it, u, states, {f"mass_{wall[j].name}": jnp.exp(log_m[i]) for i, j in enumerate(fib)}
                       | ({"beta_deg": jnp.degrees(beta)} if adapt else {}))
        if du < pre_tol and db < 1e-3:
            break
        xs, fs = (xs + [jnp.concatenate(x)])[-anderson - 1:], (fs + [jnp.concatenate(f)])[-anderson - 1:]
        y = fem.anderson(xs, fs) + momentum * step
        step = y - xs[-1]
        G_e = y[:n_G].reshape(G_e.shape)
        if adapt:
            log_m = jnp.clip(y[n_G:n_G + log_m.size].reshape(log_m.shape), np.log(0.05), np.log(20.0))
            if deposit == "iterate":
                beta = y[n_G + log_m.size:].reshape(shp)
    if pre_cache is not None and not cached:
        Path(pre_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(pre_cache, G_e=np.asarray(G_e), log_m=np.asarray(log_m), beta=np.asarray(beta))

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
