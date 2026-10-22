"""Idealized carotid bifurcation (fem.mesh.carotid_bifurcation): homeostatic prestress of model E, tortuosity after
elastin loss with the Weiss 2020 wall."""

import time

import jax.numpy as jnp
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fem
import hcmm
import materials
import plotting
from analysis import tortuosity as ti
from fem import mesh as meshlib
from fem.tensor3 import det3
from studies.common import Log, outputs, wall_stress, write_csv
from studies.model_e import artery_par, artery_simulate
from studies.weiss.model import KO, WALL_WEISS, weiss_par

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


def bifurcation(half=False, D=6.3, angles=(25.0, 25.0), n_q=6, n_seam=8, n_r=2, name=None):
    """Prestress at p_hom until u = 0 (model E, hybrid hex8, incompressible elastin); fiber directions from the Laplace
    wall basis (CCA inlet -> ICA, ECA outlets); end rings: no axial and no tangential displacement (radially free);
    half: symmetry plane y = 0 (bifurcation plane)"""
    name = name or ("bifurcation_half" if half else "bifurcation")
    log = Log()
    txt, pvd, fig_path = outputs("carotid/bifurcation", f"{name}.txt", f"{name}.pvd", f"{name}.png")
    m = meshlib.carotid_bifurcation(D=D, angles=angles, n_q=n_q, n_seam=n_seam, n_r=n_r, half=half)
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
    meshlib.VTKSeries(pvd).write(0.0, m.X, m.conn, *sysm.fields(u, states, dict(
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


def bifurcation_tortuosity(D=6.3, angles=(25.0, 25.0), lengths=(3.0, 8.0, 3.0), n_q=4, n_seam=5, n_r=2, noise=0.01,
                           ell=0.88, kappa_junction=0.10, insult_day=10.0, n_hold=400, gamma=1.0, ds=1.0,
                           record_every=10, tol=1e-10, length=None, ends="isometric", half=False, name=None, **kw):
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
    Output per step: TI, deflection and axial force of the ICA tube (cuff to the distal end)"""
    log = Log()
    name = name or "bifurcation_tortuosity"
    txt, pvd, csv_path, fig, img = outputs("carotid/tortuosity", *(f"{name}{s}" for s in (
        ".txt", ".pvd", ".csv", ".png", "_end_Jg.png")))
    par = weiss_par(**kw)
    R, p0 = 0.5 * D, float(par["p_hom"])
    kappa = float(par["h"] / par["a"])
    m = meshlib.carotid_bifurcation(D=D, angles=angles, lengths=lengths, n_q=n_q, n_seam=n_seam, n_r=n_r,
                                    kappa=kappa, kappa_junction=kappa_junction, half=half)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"], element="hybrid")
    Q = jnp.asarray(fem.wall_basis(m.X, m.conn, m.nodes["inner"], m.nodes["outer"], [m.nodes["inlet"]],
                                   [m.nodes["ica"], m.nodes["eca"]]))
    length = length or max(np.sqrt(R * kappa * R), 1.2 * 0.5 * np.pi * R / n_q)
    bc = end_bc(m, angles)(p0)
    e_ica = axes(angles)["ica"]
    ica_nodes = np.intersect1d(np.unique(m.conn[m.elsets["ica"]]), m.nodes["inner"])
    s_ica, ring = np.unique(np.round(m.X[ica_nodes] @ e_ica, 6), return_inverse=True)
    log(f"idealized carotid bifurcation (D {D} mm, ICA / ECA {angles[0]:g} / {angles[1]:g} deg, lengths {lengths} D), "
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
    C10, alpha = states.constituents[0].material.C10, par["diag+.alpha"]
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
    series = meshlib.VTKSeries(pvd)

    def record(k, u, states):
        diag = states.constituents[names.index("diag+")]
        e = np.asarray(sysm.gauss_F(u) @ Q)
        e = e / np.linalg.norm(e, axis=-2, keepdims=True)
        biaxial = np.einsum("egia,egij,egja->ega", e, sysm.total_stress(u, states), e)
        series.write(ds * k, m.X, m.conn, *sysm.fields(u, states, dict(
            sigma_tt_kPa=1e3 * biaxial[..., 1], sigma_zz_kPa=1e3 * biaxial[..., 2],
            c_e_rel=states.constituents[0].material.C10 / par["elastin.C10"], elastin_field=xi,
            alpha_diag_deg=angle(diag.material.M), alpha_dep_deg=angle(getattr(diag.production, "M_dep",
                                                                                 diag.material.M)),
            wall_stress_rel=states.sigma_w / states.sigma_w_h, J_g=det3(states.F_g),
            **{f"e_{c}": Q[..., :, i] for i, c in enumerate(("r", "theta", "z"))})))

    record(0, u, states)
    rows, t0 = [], time.time()
    n_steps = round(insult_day / ds) + n_hold
    ring_ica = m.nodes["ica"]
    for k in range(1, n_steps + 1):
        cs = list(states.constituents)
        if ds * k >= insult_day:
            cs[0] = cs[0].replace(material=materials.NeoHookeanInc(KO["C10"] * scale))
        s_tt, s_zz = in_plane(u, states)
        al = jnp.asarray(np.arctan(np.tan(alpha) * (np.maximum(s_tt, 0.0) / np.maximum(s_zz, 1e-12) / ratio_h)
                                   ** gamma))[..., None]
        for d, sign in (("diag+", 1.0), ("diag-", -1.0)):
            c = cs[names.index(d)]
            cs[names.index(d)] = c.replace(production=c.production.replace(M_dep=jnp.sin(al) * e_t
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
        J_g = np.asarray(det3(states.F_g))
        al_fib = np.asarray(angle(states.constituents[names.index("diag+")].material.M))
        rows.append(dict(day=ds * k, TI_ica=ti(c), deflection_ica=np.linalg.norm(off, axis=1).max() / R,
                         J_g_mean=J_g.mean(), J_g_max=J_g.max(), alpha_mean=al_fib.mean(), alpha_min=al_fib.min(),
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
    outer_image(m, m.X + np.asarray(u).reshape(-1, 3), np.asarray(det3(states.F_g)).mean(1), img, "J_g",
                f"{name}, day {ds * n_steps:g}")
    log.save(txt)
