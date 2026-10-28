"""Idealized carotid bifurcation (fem.mesh.carotid_bifurcation): homeostatic prestress of model E, tortuosity after
elastin loss with the Weiss 2020 wall."""

import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fcmm
import fem
import hcmm
import materials
import plotting
from analysis import tortuosity as ti
from fem import mesh as meshlib
from fem.tensor3 import inv3
from rod import XG, Rod
from studies.common import Log, outputs, wall_stress, write_csv
from studies.model_e import artery_par, artery_simulate
from studies.weiss.model import KO, WALL_WEISS, weiss_par
from wall import assemble

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GEOMETRY = {  # CCA diameter D (mm); sinus (proximal ICA), distal ICA, ECA diameters / D; ICA, ECA angles from the CCA (deg)
    "bharadvaj": dict(D=6.3, sinus=1.11, ica=0.71, eca=0.57, angles=(25.0, 25.0)),
    "goubergrits": dict(D=6.45, sinus=1.09, ica=0.80, eca=0.72, angles=(33.8, 27.8)),
}


def carotid_geometry(geometry, **override):
    """GEOMETRY[geometry] with the values of override that are not None; "bharadvaj": Bharadvaj et al. 1982 /
    Ku et al. 1985 (fem.mesh.carotid_bifurcation defaults), "goubergrits": means of 86 casts at 80 mmHg
    (Goubergrits et al. 2002, Table 3: D_c 6.45, D_i1 7.04, D_i2 5.17, D_e 4.63 mm, AI 33.8, AE 27.8 deg;
    SD 15-20 % of the diameters, 12 deg of the angles)"""
    return GEOMETRY[geometry] | {k: v for k, v in override.items() if v is not None}


def axes(angles):
    """unit axes of CCA (inlet), ICA, ECA"""
    a, b = np.radians(angles)
    return {"inlet": np.array([0.0, 0, 1]), "ica": np.array([np.sin(a), 0, np.cos(a)]),
            "eca": np.array([-np.sin(b), 0, np.cos(b)])}


def end_bc(m, angles, axial=("inlet", "ica", "eca")):
    """end rings: no tangential displacement (no twist, no lateral motion), no axial displacement for the rings in
    axial, radially free; half: u_y = 0 on y = 0"""
    nodes, normals = [], []
    for ring, e in axes(angles).items():
        i = m.nodes[ring]
        d = m.X[i] - m.X[i].mean(0)
        d -= (d @ e)[:, None] * e
        nodes += [i, i] if ring in axial else [i]
        normals += ([np.tile(e, (len(i), 1))] if ring in axial else []) + [
            np.cross(e, d / np.linalg.norm(d, axis=1, keepdims=True))]
    slip = (np.concatenate(nodes), np.concatenate(normals))
    fixed = 3 * m.nodes["sym"] + 1 if "sym" in m.nodes else np.zeros(0, int)
    return lambda p, f_dead=None: fem.BC(fixed, np.zeros(len(fixed)), p=p, slip=slip, f_dead=f_dead)


def outer_image(m, x, val, path, label, title=""):
    """outer surface at nodal positions x coloured by an element value: bifurcation plane and 3D view"""
    faces = m.faces["outer"]
    owner = {tuple(sorted(f)): e for e, f in zip(*meshlib.boundary_faces(m)[::2])}
    val = val[[owner[tuple(sorted(f))] for f in faces]]
    fig = plt.figure(figsize=(12, 7))
    norm = plt.Normalize(np.percentile(val, 1), np.percentile(val, 99))
    for p, (elev, azim, sub) in enumerate([(90, -90, "front view (bifurcation plane)"), (20, -55, "3D")]):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(x[faces][:, :, [0, 2, 1]], facecolors=plt.cm.viridis(norm(val)),
                                             edgecolors="k", linewidths=0.1))
        lo, hi = x.min(0), x.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
        ax.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)"); ax.set_title(sub)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=fig.axes, shrink=0.5, label=label)
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bifurcation(half=False, geometry="goubergrits", D=None, angles=None, sinus=None, ica=None, eca=None, n_q=6, n_seam=8,
                n_r=2, name=None):
    """Prestress at p_hom until u = 0 (model E, hybrid hex8, incompressible elastin); fiber directions from the Laplace
    wall basis (CCA inlet -> ICA, ECA outlets); end rings: no axial and no tangential displacement (radially free);
    half: symmetry plane y = 0 (bifurcation plane); geometry: carotid_geometry preset, D ... eca override it"""
    g = carotid_geometry(geometry, D=D, angles=angles, sinus=sinus, ica=ica, eca=eca)
    D, angles = g["D"], g["angles"]
    name = name or ("bifurcation_half" if half else "bifurcation")
    log = Log()
    txt, pvd, fig_path = outputs("carotid/bifurcation", f"{name}.txt", f"{name}.pvd", f"{name}.png")
    m = meshlib.carotid_bifurcation(D=D, angles=angles, sinus=g["sinus"], ica=g["ica"], eca=g["eca"], n_q=n_q,
                                    n_seam=n_seam, n_r=n_r, half=half)
    _, wdet = fem.geometry(m.X, m.conn)
    Q = fem.wall_basis(m.X, m.conn, m.nodes["inner"], m.nodes["outer"], [m.nodes["inlet"]],
                       [m.nodes["ica"], m.nodes["eca"]])
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"], element="hybrid")
    log(f"idealized carotid bifurcation, D = {D} mm, ICA / ECA at {angles[0]:g} / {angles[1]:g} deg"
        f"{', half (y >= 0)' if half else ''}: {m.n_elem} hex8 (hybrid), {sysm.n_dof} dofs; "
        + ", ".join(f"{k} {len(v)}" for k, v in m.elsets.items()))
    log(f"  element quality: min det J / max det J per element {(wdet.min(1) / wdet.max(1)).min():.3f}, "
        f"inverted {(wdet.min(1) <= 0).sum()}; basis orthonormality "
        f"{np.abs(np.swapaxes(Q, -1, -2) @ Q - np.eye(3)).max():.1e}")

    bc = end_bc(m, angles)
    par = artery_par()
    Qj = jnp.asarray(Q)
    t0 = time.time()
    _, u, states = artery_simulate(m, sysm, Qj, par, n_steps=0, bc=bc, final=True, incompressible=True,
                                   measure=lambda u: (jnp.zeros(()),), log=log, anderson=2, pre_iter=200)
    log(f"  prestress at p_hom = {1e3 * float(par['p_hom']):.0f} kPa done in {time.time() - t0:.0f}s")

    s = sysm.stress(u, states)
    s_loc = np.asarray(1e3 * (jnp.swapaxes(Qj, -1, -2) @ s @ Qj))
    dev = s - jnp.trace(s, axis1=-2, axis2=-1)[..., None, None] / 3 * jnp.eye(3)
    mises = np.asarray(1e3 * jnp.sqrt(1.5 * jnp.sum(dev ** 2, (-2, -1))))
    meshlib.VTKSeries(pvd, m).write(0.0, m.X, m.conn, *sysm.fields(u, states, dict(
        sigma_tt_kPa=s_loc[..., 1, 1], sigma_zz_kPa=s_loc[..., 2, 2], mises_kPa=mises,
        **{f"e_{k}": Q[..., i] for i, k in enumerate(("r", "theta", "z"))})))
    log("  homeostatic stress (kPa), mean / min / max over the Gauss points:")
    for reg, el in m.elsets.items():
        log(f"    {reg:9s} sigma_tt {s_loc[el, :, 1, 1].mean():6.1f} / {s_loc[el, :, 1, 1].min():6.1f} / "
            f"{s_loc[el, :, 1, 1].max():6.1f}   von Mises {mises[el].mean():6.1f} / {mises[el].min():6.1f} / "
            f"{mises[el].max():6.1f}")
    log(f"  ParaView: {pvd.name}")
    outer_image(m, m.X, mises.mean(1), fig_path, "homeostatic von Mises stress (kPa), outer surface")
    log.save(txt)


def bifurcation_tortuosity(geometry="goubergrits", D=None, angles=None, sinus=None, ica=None, eca=None,
                           lengths=(3.0, 8.0, 3.0), n_q=4, n_seam=5, n_r=2, noise=0.01,
                           ell=0.88, kappa_junction=0.10, insult_day=10.0, n_hold=400, gamma=1.0, ds=1.0,
                           record_every=10, tol=1e-10, length=None, ends="isometric", half=False, model="hcmm",
                           fcmm_prestress=200, name=None, **kw):
    """Weiss 2020 wall (table S2) on the idealized bifurcation, wall thickness h / a of Weiss in the tubes (the
    mechanics are scale-free, so this is the mouse wall on a scaled carotid), kappa_junction r on the junction patches
    (with h / a = 0.06 there the prestress stalls; a thicker wall everywhere turns the response to the elastin loss
    into mass loss), prestressed at 120 mmHg (end_bc: isometric ends); ends "isometric" (kept) or "isotonic" (CCA
    inlet kept, ICA and ECA ends axially free under the axial end reactions of the prestressed state as dead loads);
    on insult_day elastin stiffness c_e -> KO value everywhere in one step, then n_hold steps of ds days;
    diagonal collagen deposited at the emergent angle (Hariton 2007, as weiss_tortuosity orient="stress"):
        tan alpha_dep = tan alpha_h ((s_tt / s_zz) / (s_tt / s_zz)_h)^gamma,   s_ii = sigma_ii - sigma_rr
    in the Laplace wall basis (theta, z); classical kinematics (growth "fixed"), tr sigma stimulus, nonlocal over
    length (default max(sqrt(R h), 1.2 x element size): a mean shorter than the elements is pointwise and localizes); symmetry breaking: c_e (1 + noise xi(X)), xi a fixed 3D Gaussian random field (seed 0, 256 random
    Fourier modes, correlation length ell R, max|xi| = 1), already in the prestress.
    model "fcmm": full constrained mixture (cohorts over the window of T, each diagonal cohort along its deposition
    angle, alpha output: mass-weighted mean direction), from the HCMM prestress with fcmm_prestress further prestress
    iterations (1: equilibrium only, the FCMM cohort quadrature leaves a small u; the G&R starts there).
    geometry: carotid_geometry preset, D ... eca override it.
    Output per step: TI, deflection and axial force of the ICA tube (cuff to the distal end)"""
    g = carotid_geometry(geometry, D=D, angles=angles, sinus=sinus, ica=ica, eca=eca)
    D, angles = g["D"], g["angles"]
    log = Log()
    name = name or "bifurcation_tortuosity"
    txt, pvd, csv_path, fig, img = outputs("carotid/tortuosity", *(f"{name}{s}" for s in (
        ".txt", ".pvd", ".csv", ".png", "_end_Jg.png")))
    par = weiss_par(**kw)
    R, p0 = 0.5 * D, float(par["p_hom"])
    kappa = float(par["h"] / par["a"])
    m = meshlib.carotid_bifurcation(D=D, angles=angles, sinus=g["sinus"], ica=g["ica"], eca=g["eca"], lengths=lengths,
                                    n_q=n_q, n_seam=n_seam, n_r=n_r, kappa=kappa, kappa_junction=kappa_junction,
                                    half=half)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"], element="hybrid")
    Q = jnp.asarray(fem.wall_basis(m.X, m.conn, m.nodes["inner"], m.nodes["outer"], [m.nodes["inlet"]],
                                   [m.nodes["ica"], m.nodes["eca"]]))
    length = length or max(np.sqrt(R * kappa * R), 1.2 * 0.5 * np.pi * R / n_q)
    bc = end_bc(m, angles)(p0)
    e_ica = axes(angles)["ica"]
    ica_nodes = np.intersect1d(np.unique(m.conn[m.elsets["ica"]]), m.nodes["inner"])
    s_ica, ring = np.unique(np.round(m.X[ica_nodes] @ e_ica, 6), return_inverse=True)
    log(f"idealized carotid bifurcation ({geometry}: D {D} mm, ICA / ECA {angles[0]:g} / {angles[1]:g} deg, diameters "
        f"sinus / ICA / ECA {g['sinus']} / {g['ica']} / {g['eca']} D, lengths {lengths} D), "
        f"h = {kappa:.3f} r (Weiss), {kappa_junction} r in the junction: {m.n_elem} hex8 (hybrid), {sysm.n_dof} dofs; Weiss table S2 wall, tr sigma "
        f"stimulus (nonlocal {length:.2f} mm), growth fixed, emergent collagen angle (gamma {gamma:g}); elastin c_e -> KO "
        f"on day {insult_day:g}, {n_hold} days held; elastin field {noise:g}; {ends} ends; ICA tube {len(s_ica)} rings")

    xg = np.asarray(fem.gauss_points(m.X, m.conn))
    rng = np.random.default_rng(0)
    w, ph = rng.standard_normal((256, 3)) / (ell * R), rng.uniform(0, 2 * np.pi, 256)
    xi = np.cos(xg @ w.T + ph).sum(-1)
    xi = (xi - xi.mean()) / np.abs(xi - xi.mean()).max()
    scale = jnp.asarray(1.0 + noise * xi)
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = Q @ jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])) @ jnp.swapaxes(Q, -1, -2)
    t0 = time.time()
    _, u, states = artery_simulate(m, sysm, Q, par, n_steps=0, bc=lambda p: bc, final=True, log=log, ds=ds,
                                   wall=WALL_WEISS, laws="wall", incompressible=True, growth="fixed", G_e=G_e,
                                   measure=lambda u: (jnp.zeros(()),),
                                   fields={"elastin.C10": par["elastin.C10"] * scale} if noise else None,
                                   anderson=2, pre_iter=200)
    log(f"  prestress done in {time.time() - t0:.0f}s")
    fc = model == "fcmm"
    if fc:
        T = max(float(v) for k, v in par.items() if k.endswith(".T"))
        n_max = fcmm.window_for(T, ds) + 2
        sysm = fem.System(m, fcmm, pressure_faces=m.faces["inner"], element="hybrid")
        t0 = time.time()
        _, u, states = artery_simulate(m, sysm, Q, par, n_steps=0, bc=lambda p: bc, final=True, log=log, ds=ds,
                                       wall=WALL_WEISS, laws="wall", incompressible=True, growth="fixed",
                                       G_e=inv3(states.constituents[0].F_r), measure=lambda u: (jnp.zeros(()),),
                                       fields={"elastin.C10": par["elastin.C10"] * scale} if noise else None,
                                       anderson=2, pre_iter=fcmm_prestress, n_max=n_max)
        u = jnp.asarray(sysm.last_u)
        log(f"  FCMM ({n_max} cohorts, T {T:g} d) from the HCMM elastin state, {fcmm_prestress} prestress iteration(s): "
            f"max|u| {float(jnp.abs(u).max()):.2e} mm  ({time.time() - t0:.0f}s)")
    part = lambda c: c.params if fc else c
    set_part = lambda c, **f: c.replace(params=c.params.replace(**f)) if fc else c.replace(**f)
    J_g = lambda st: np.asarray(jax.vmap(jax.vmap(sysm.J_target))(st))

    def fiber(c):
        """fiber direction; FCMM: mass-weighted mean over the cohorts"""
        if not fc:
            return c.material.M
        h = c.history
        w = (h.m * jnp.exp(h.K_cumu - jnp.take_along_axis(h.K_cumu, h.n[..., None], -1))
             * (jnp.arange(h.m.shape[-1]) <= h.n[..., None]))
        M = jnp.einsum("...k,...ki->...i", w, h.material.M)
        return M / jnp.linalg.norm(M, axis=-1, keepdims=True)

    states = states.replace(sigma_w=(sw := wall_stress(sysm, u, states, length, None, True)), sigma_w_h=sw)
    internal = lambda u, states: np.asarray(sysm.residual(jnp.concatenate([jnp.asarray(u), jnp.asarray(sysm.last_p)]),
                                                          states, p0, jnp.zeros(sysm.n_dof)))[:sysm.n_dof].reshape(-1, 3)
    if ends == "isotonic":
        r, f_dead = internal(u, states), np.zeros((m.n_nodes, 3))
        for end in ("ica", "eca"):
            e = axes(angles)[end]
            f_dead[m.nodes[end]] = (r[m.nodes[end]] @ e)[:, None] * e
        bc = end_bc(m, angles, axial=("inlet",))(p0, f_dead.ravel())
        log("  isotonic: axial end forces " + ", ".join(
            f"{end} {1e3 * float(f_dead[m.nodes[end]].sum(0) @ axes(angles)[end]):.1f} mN" for end in ("ica", "eca")))

    names = [q.name for q in WALL_WEISS]
    alpha = par["diag+.alpha"]
    e_t, e_z = Q[..., :, 1], Q[..., :, 2]
    angle = lambda M: jnp.degrees(jnp.arccos(jnp.abs(jnp.sum(M * e_z, axis=-1))))

    def in_plane(u, states):
        """(s_tt - s_rr, s_zz - s_rr) along the deformed wall directions, nonlocal mean"""
        F, sig = sysm.gauss_F(u), sysm.stress(u, states)
        s = lambda n: jnp.einsum("...i,...ij,...j->...", n, sig, n) / jnp.einsum("...i,...i->...", n, n)
        rr = s(jnp.linalg.solve(jnp.swapaxes(F, -1, -2), Q[..., :, 0][..., None])[..., 0])
        return [sysm.nonlocal_mean(s(jnp.einsum("...ij,...j->...i", F, Q[..., :, i])) - rr, length) for i in (1, 2)]

    s_tt, s_zz = in_plane(u, states)
    ratio_h = s_tt / s_zz
    series = meshlib.VTKSeries(pvd, m)

    def record(k, u, states):
        diag = states.constituents[names.index("diag+")]
        e = np.asarray(sysm.gauss_F(u) @ Q)
        e = e / np.linalg.norm(e, axis=-2, keepdims=True)
        biaxial = np.einsum("egia,egij,egja->ega", e, sysm.total_stress(u, states), e)
        series.write(ds * k, m.X, m.conn, *sysm.fields(u, states, dict(
            sigma_tt_kPa=1e3 * biaxial[..., 1], sigma_zz_kPa=1e3 * biaxial[..., 2],
            c_e_rel=part(states.constituents[0]).material.C10 / par["elastin.C10"], elastin_field=xi,
            alpha_diag_deg=angle(fiber(diag)), alpha_dep_deg=angle(part(diag).production.M_dep),
            wall_stress_rel=states.sigma_w / states.sigma_w_h, J_g=J_g(states),
            **{f"e_{c}": Q[..., :, i] for i, c in enumerate(("r", "theta", "z"))})))

    record(0, u, states)
    rows, t0 = [], time.time()
    n_steps = round(insult_day / ds) + n_hold
    ring_ica = m.nodes["ica"]
    for k in range(1, n_steps + 1):
        cs = list(states.constituents)
        if ds * k >= insult_day:
            cs[0] = set_part(cs[0], material=materials.NeoHookeanInc(KO["C10"] * scale))
        s_tt, s_zz = in_plane(u, states)
        al = jnp.asarray(np.arctan(np.tan(alpha) * (np.maximum(s_tt, 0.0) / np.maximum(s_zz, 1e-12) / ratio_h)
                                   ** gamma))[..., None]
        for d, sign in (("diag+", 1.0), ("diag-", -1.0)):
            c = cs[names.index(d)]
            cs[names.index(d)] = set_part(c, production=part(c).production.replace(M_dep=jnp.sin(al) * e_t
                                                                                   + sign * jnp.cos(al) * e_z))
        states = states.replace(constituents=cs, sigma_w=wall_stress(sysm, u, states, length, None, True))
        u, states, _ = sysm.run(states, [bc], u0=u, tol=tol)
        x = m.X + np.asarray(u).reshape(-1, 3)
        c = np.stack([np.bincount(ring, x[ica_nodes, i]) / np.bincount(ring) for i in range(3)], -1)
        if half:
            c[:, 1] = 0.0
        chord = (c[-1] - c[0]) / np.linalg.norm(c[-1] - c[0])
        off = (c - c[0]) - ((c - c[0]) @ chord)[:, None] * chord
        r = internal(u, states)
        jg = J_g(states)
        al_fib = np.asarray(angle(fiber(states.constituents[names.index("diag+")])))
        rows.append(dict(day=ds * k, TI_ica=ti(c), deflection_ica=np.linalg.norm(off, axis=1).max() / R,
                         J_g_mean=jg.mean(), J_g_max=jg.max(), alpha_mean=al_fib.mean(), alpha_min=al_fib.min(),
                         alpha_max=al_fib.max(), axial_force_ica_mN=-1e3 * float(r[ring_ica].sum(0) @ e_ica),
                         ica_end_shift_mm=float(np.asarray(u).reshape(-1, 3)[ring_ica].mean(0) @ e_ica)))
        if k % record_every == 0 or k == n_steps:
            record(k, u, states)
        if k % max(n_steps // 20, 1) == 0:
            q = rows[-1]
            log(f"  day {q['day']:5.0f}  ICA: TI {q['TI_ica']:.2e}  deflection/R {q['deflection_ica']:.3f}  axial force "
                f"{q['axial_force_ica_mN']:.1f} mN  end shift {q['ica_end_shift_mm']:+.2f} mm   J_g mean {q['J_g_mean']:.4f} max {q['J_g_max']:.4f}   alpha mean "
                f"{q['alpha_mean']:.2f} ({q['alpha_min']:.2f}-{q['alpha_max']:.2f}) deg  ({time.time() - t0:.0f}s)")
    log(f"  ParaView: {pvd.name} ({len(series.entries)} time steps)")
    write_csv(rows, csv_path)
    col = lambda key: np.array([q[key] for q in rows])
    t = col("day")
    plotting.panels([dict(t=t, series={"ICA": col("TI_ica")}, title="Tortuosity index", ylabel="(-)"),
                     dict(t=t, series={"ICA": col("deflection_ica")}, title="Lateral deflection / R", ylabel="(-)"),
                     dict(t=t, series={"ICA": col("axial_force_ica_mN")}, title="Axial force", ylabel="mN"),
                     dict(t=t, series={"mean": col("J_g_mean"), "max": col("J_g_max")}, title="Volume growth J_g",
                          ylabel="(-)"),
                     dict(t=t, series={"mean": col("alpha_mean"), "min": col("alpha_min"), "max": col("alpha_max")},
                          title="Diagonal collagen angle", ylabel="deg")], fig)
    outer_image(m, m.X + np.asarray(u).reshape(-1, 3), J_g(states).mean(1), img, "J_g",
                f"{name}, day {ds * n_steps:g}")
    log.save(txt)


def bifurcation_rod(geometry="goubergrits", D=None, angles=None, sinus=None, ica=None, eca=None, lengths=(3.0, 8.0, 3.0),
                    s_sinus=(0.3, 0.8, 1.8), cuffs=(0.6, 1.5, 1.4), kappa_junction=0.10, tie_EA=2.0, el=0.5,
                    n_ring=16, noise=0.01,
                    ell=0.88, insult_day=10.0, n_hold=400, gamma=1.0, ds=1.0, length=None, record_every=10, name=None,
                    **kw):
    """bifurcation_tortuosity on the reduced model (rod.Rod): CCA, ICA, ECA centerlines as 3D rods of elements of
    about el mm, joined rigidly at the origin, ends clamped (isometric); radii of fem.mesh.carotid_bifurcation (ICA
    0.9 R -> sinus R on [0, s1], sinus until s2, -> ica R at s3; ECA eca R), wall thickness h / a of Weiss, as in the
    FE kappa_junction r from the junction to the cuffs (cuffs D along CCA, ICA, ECA; blended over 0.2 D): the
    junction carries the pressure resultant of the three lumens, with the tube wall there its sections bend and grow
    without bound (J_g > 20 by day 380); junction element: a triangle of elastic bars (tie_EA in N, 0: none) between the
    branch centres at the cuffs, the flow divider (ICA-ECA) and the lateral walls (CCA-ICA, CCA-ECA) that hold the Y
    together in tension, prestressed with the elastin (rod.Rod.tie);
    elastin c_e -> KO on insult_day, n_hold steps of ds days; emergent collagen angle (gamma); tr sigma stimulus,
    nonlocal over length (default max(sqrt(R h), 1.2 x ring point spacing)); the same 3D elastin field as the FE
    (seed 0, 256 Fourier modes, correlation length ell R). Output per step: TI, deflection and axial force of the
    ICA beyond its cuff, J_g, alpha; ParaView: deformed walls of the three branches; geometry: carotid_geometry
    preset, D ... eca override it."""
    g = carotid_geometry(geometry, D=D, angles=angles, sinus=sinus, ica=ica, eca=eca)
    D, angles, sinus, ica, eca = g["D"], g["angles"], g["sinus"], g["ica"], g["eca"]
    log = Log()
    name = name or "bifurcation_rod"
    txt, pvd, csv_path, fig = outputs("carotid/rod", *(f"{name}{s}" for s in (".txt", ".pvd", ".csv", ".png")))
    par = weiss_par(**kw)
    R, p0 = 0.5 * D, float(par["p_hom"])
    kappa = float(par["h"] / par["a"])
    length = length or max(np.sqrt(R * kappa * R), 1.2 * 2 * np.pi * R / n_ring)
    e = axes(angles)
    X, conn, branch = [np.zeros(3)], [], []
    for b, (key, L_b) in enumerate(zip(("inlet", "ica", "eca"), lengths)):
        n = max(2, round(L_b * D / el))
        i0 = len(X)
        if key == "inlet":
            X += [-L_b * D * (1 - i / n) * e[key] for i in range(n)]
            conn += [(i0 + i, i0 + i + 1) for i in range(n - 1)] + [(i0 + n - 1, 0)]
        else:
            X += [L_b * D * (i + 1) / n * e[key] for i in range(n)]
            conn += [(0, i0)] + [(i0 + i, i0 + i + 1) for i in range(n - 1)]
        branch += [b] * n
    X, conn, branch = np.array(X), np.array(conn), np.array(branch)
    ends = [1, conn[branch == 1][-1, 1], conn[branch == 2][-1, 1]]
    xs = X[conn[:, 0]][:, None] + XG[None, :, None] * (X[conn[:, 1]] - X[conn[:, 0]])[:, None]
    s1, s2, s3 = (v * D for v in s_sinus)
    sm = lambda s, lo, hi: (lambda t: t * t * (3 - 2 * t))(np.clip((s - lo) / (hi - lo), 0.0, 1.0))
    keys = ("inlet", "ica", "eca")
    e_b = np.array([e[k] for k in keys])[branch]
    radius = lambda x: np.select([branch[:, None] == 0, branch[:, None] == 1], [
        np.full(x.shape[:2], R),
        0.9 * R + (sinus - 0.9) * R * sm(x @ e["ica"], 0, s1) + (ica - sinus) * R * sm(x @ e["ica"], s2, s3)], eca * R)
    A_end, A = radius(X[conn]), radius(xs)
    s = np.abs(np.einsum("egi,ei->eg", xs, e_b))
    s_cuff = D * np.array(cuffs)[branch][:, None]
    kap = kappa_junction + (kappa - kappa_junction) * sm(s, s_cuff - 0.2 * D, s_cuff)
    rod = Rod(X, conn, branch, A_end, kap * A, hcmm, rigid=(0,), clamped=ends, n_ring=n_ring)
    if tie_EA:
        nodes_b = [np.setdiff1d(conn[branch == b].ravel(), [0]) for b in range(3)]
        cuff = [n[np.argmin(np.abs(np.abs(X[n] @ e[k]) - c * D))] for n, k, c in zip(nodes_b, keys, cuffs)]
        rod.tie([(cuff[1], cuff[2]), (cuff[0], cuff[1]), (cuff[0], cuff[2])], tie_EA)
    shp = (3 * len(conn), n_ring)
    rng = np.random.default_rng(0)
    w, ph = rng.standard_normal((256, 3)) / (ell * R), rng.uniform(0, 2 * np.pi, 256)
    xi = np.cos(rod.X_points @ w.T + ph).sum(-1)
    xi = ((xi - xi.mean()) / np.abs(xi - xi.mean()).max()).reshape(shp)
    scale = jnp.asarray(1.0 + noise * xi)
    ica_nodes = np.concatenate([[0], conn[branch == 1][:, 1]])
    ica_nodes = ica_nodes[X[ica_nodes] @ e["ica"] >= cuffs[1] * D]
    log(f"idealized carotid bifurcation as 3D rods ({geometry}: D {D} mm, ICA / ECA {angles[0]:g} / {angles[1]:g} deg, lengths "
        f"{lengths} D, radii ICA 0.9-{sinus}-{ica} R, ECA {eca} R), h = {kappa:.3f} r (Weiss), {kappa_junction} r "
        f"up to the cuffs {cuffs} D: "
        f"{', '.join(f'{k} {np.sum(branch == b)}' for b, k in enumerate(('CCA', 'ICA', 'ECA')))} elements x 3 sections "
        f"x {n_ring} ring points, {len(rod.free)} DOFs; rigid junction, clamped isometric ends; tr sigma stimulus "
        f"(nonlocal {length:.2f} mm), emergent collagen angle (gamma {gamma:g}); c_e -> KO on day {insult_day:g}, "
        f"{n_hold} days held; elastin field {noise:g}")

    t0 = time.time()
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = jnp.broadcast_to(jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])), shp + (3, 3))
    Q = jnp.broadcast_to(jnp.eye(3), G_e.shape)
    states, it, du = rod.prestress(lambda G: assemble(G, Q, par, WALL_WEISS, ds=ds, laws="wall", incompressible=True,
                                                      growth="fixed", fields={"elastin.C10": par["elastin.C10"] * scale}),
                                   G_e, p0)
    log(f"  prestress: {it} iterations, max|q - q_0| {du:.1e}  ({time.time() - t0:.0f}s)"
        + (f"; junction ties ICA-ECA / CCA-ICA / CCA-ECA {', '.join(f'{1e3 * v:.1f}' for v in rod.tie_forces(rod.q0))} mN"
           if tie_EA else ""))
    q = rod.q0
    s_tt, s_zz = rod.stresses(q, states)
    sw = rod.nonlocal_mean(s_tt + s_zz, length)
    states = states.replace(sigma_w=sw, sigma_w_h=sw)
    ratio_h = rod.nonlocal_mean(s_tt, length) / rod.nonlocal_mean(s_zz, length)
    names = [c.name for c in WALL_WEISS]
    alpha = par["diag+.alpha"]
    e_t, e_z = jnp.array([0.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    angle = lambda M: np.degrees(np.arccos(np.abs(np.asarray(M @ e_z))))
    J_g = lambda st: np.asarray(jax.vmap(jax.vmap(hcmm.J_target))(st))
    series = meshlib.VTKSeries(pvd)
    X_ref = rod.wall(q, states)
    last_ica = 3 * int(np.flatnonzero(branch == 1)[-1]) + 2

    def record(day, q, st):
        Xw, cw = rod.wall(q, st)
        s_tt, s_zz = (np.asarray(v) for v in rod.stresses(q, st))
        diag = st.constituents[names.index("diag+")]
        per = lambda v: np.repeat(np.asarray(v).reshape(-1), 2)
        fields = dict(J_g=per(J_g(st)), sigma_tt_kPa=per(1e3 * s_tt), sigma_zz_kPa=per(1e3 * s_zz),
                      c_e_rel=per(st.constituents[0].material.C10 / par["elastin.C10"]), elastin_field=per(xi),
                      alpha_diag_deg=per(angle(diag.material.M)), alpha_dep_deg=per(angle(diag.production.M_dep)),
                      wall_stress_rel=per(st.sigma_w / st.sigma_w_h), branch=per(np.repeat(branch, 3 * n_ring)))
        series.write(day, X_ref[0], cw, fields | {"u": Xw - X_ref[0]}, {k: v[cw].mean(1) for k, v in fields.items()})

    record(0.0, q, states)
    rows, t0 = [], time.time()
    n_steps = round(insult_day / ds) + n_hold
    for k in range(1, n_steps + 1):
        cs = list(states.constituents)
        if ds * k >= insult_day:
            cs[0] = cs[0].replace(material=materials.NeoHookeanInc(KO["C10"] * scale))
        s_tt, s_zz = rod.stresses(q, states)
        r_tt, r_zz = rod.nonlocal_mean(s_tt, length), rod.nonlocal_mean(s_zz, length)
        al = jnp.arctan(jnp.tan(alpha) * (jnp.maximum(r_tt, 0.0) / jnp.maximum(r_zz, 1e-12) / ratio_h) ** gamma)
        for d, sign in (("diag+", 1.0), ("diag-", -1.0)):
            c = cs[names.index(d)]
            cs[names.index(d)] = c.replace(production=c.production.replace(
                M_dep=jnp.sin(al)[..., None] * e_t + sign * jnp.cos(al)[..., None] * e_z))
        states = states.replace(constituents=cs, sigma_w=rod.nonlocal_mean(s_tt + s_zz, length))
        q, it = rod.solve(q, states, p0)
        states = rod.commit(q, states)
        c = rod.positions(q)[ica_nodes]
        chord = (c[-1] - c[0]) / np.linalg.norm(c[-1] - c[0])
        off = (c - c[0]) - ((c - c[0]) @ chord)[:, None] * chord
        F = np.asarray(rod.F(q, states))
        a_end = float(np.asarray(rod.expand(q))[last_ica // 3, 12:14] @ np.array([1 - XG[2], XG[2]]))
        s_zz = np.asarray(rod.stresses(q, states)[1])
        N = float(np.sum(s_zz[last_ica] * kappa * A.ravel()[last_ica] * F[last_ica, :, 0, 0] * a_end * 2 * np.pi
                         / n_ring))
        jg, al_f = J_g(states), angle(states.constituents[names.index("diag+")].material.M)
        rows.append(dict(day=ds * k, TI_ica=ti(c), deflection_ica=np.linalg.norm(off, axis=1).max() / R,
                         out_of_plane_ica=np.abs(off[:, 1]).max() / R, J_g_mean=jg.mean(), J_g_max=jg.max(),
                         alpha_mean=al_f.mean(), alpha_min=al_f.min(), alpha_max=al_f.max(), axial_force_ica_mN=1e3 * N,
                         **({f"tie_{t}_mN": 1e3 * v for t, v in zip(("divider", "cca_ica", "cca_eca"),
                                                                    rod.tie_forces(q))} if tie_EA else {})))
        if k % record_every == 0 or k == n_steps:
            record(ds * k, q, states)
        if k % max(n_steps // 20, 1) == 0:
            r = rows[-1]
            log(f"  day {r['day']:5.0f}  ICA: TI {r['TI_ica']:.2e}  deflection/R {r['deflection_ica']:.3f} (out of plane "
                f"{r['out_of_plane_ica']:.3f})  axial force {r['axial_force_ica_mN']:.1f} mN   J_g mean "
                f"{r['J_g_mean']:.4f} max {r['J_g_max']:.4f}   alpha mean {r['alpha_mean']:.2f} ({r['alpha_min']:.2f}-"
                f"{r['alpha_max']:.2f}) deg" + (f"  ties {r['tie_divider_mN']:.0f} / {r['tie_cca_ica_mN']:.0f} / "
                                                f"{r['tie_cca_eca_mN']:.0f} mN" if tie_EA else "")
                + f"  (Newton {it}, {time.time() - t0:.0f}s)")
    log(f"  ParaView: {pvd.name} ({len(series.entries)} time steps)")
    write_csv(rows, csv_path)
    col = lambda key: np.array([r[key] for r in rows])
    t = col("day")
    plotting.panels([dict(t=t, series={"ICA": col("TI_ica")}, title="Tortuosity index", ylabel="(-)"),
                     dict(t=t, series={"ICA": col("deflection_ica")}, title="Lateral deflection / R", ylabel="(-)"),
                     dict(t=t, series={"ICA": col("axial_force_ica_mN")}, title="Axial force", ylabel="mN"),
                     dict(t=t, series={"mean": col("J_g_mean"), "max": col("J_g_max")}, title="Volume growth J_g",
                          ylabel="(-)"),
                     dict(t=t, series={"mean": col("alpha_mean"), "min": col("alpha_min"), "max": col("alpha_max")},
                          title="Diagonal collagen angle", ylabel="deg")], fig)
    log.save(txt)
