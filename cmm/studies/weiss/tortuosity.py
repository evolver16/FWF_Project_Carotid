"""Weiss et al. 2020, Fig. 5/6: tortuosity of a straight aorta from a local loss of elastic fiber integrity (Eq. 8),
isometric vs isotonic ends."""

import pathlib
import time
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates

import fem
import hcmm
import materials
import plotting
from analysis import centerline, tortuosity
from fem import mesh as meshlib
from fem.tensor3 import det3
from studies.common import Log, outputs, wall_stress, write_csv
from studies.model_e import artery_simulate
from studies.weiss.model import KO, WALL_WEISS, weiss_par


def weiss_tortuosity(ends="isometric", insult="both", local=True, tether=0.0, tether_z=(0.25, 0.75), tether_day=0.0,
                     tether_days=np.inf,
                     insult_day=0.0, orient="prescribed", gamma=1.0, noise=0.01, n=(2, 16, 120), L=14.0, n_ramp=200,
                     n_hold=0, ds=1.0, growth="fixed", stimulus="trace", length=None, record_every=10, name=None, **kw):
    """Straight half tube (in vivo a, h, length L = 2 l_o ~ 12 diameters as in Weiss Fig. 5/6, hybrid hex8), wall of
    table S2 prestressed at 120 mmHg with
    fixed ends (elastin from G_e = diag(1/(G_theta G_z), G_theta, G_z), stress-free state fixed); insult ramped over
    n_ramp steps of ds days (slow against T: quasi-static, compared over dv / dv_KO as Weiss' equilibrated states),
    then held n_hold steps (Eq. 8); pressure ramped with the insult from p_hom to p_gr (kw, MPa; default both 120 mmHg):
        v(x) = v_WT + r (v_KO - v_WT) (1 + f(x)) / 2,   r = clip((day - insult_day) / (n_ramp ds), 0, 1)
        f    = 1/2 (1 - cos(2 pi z / l_o)) exp(-((theta - theta_hat) / 2)^2),   theta_hat = pi (z < l_o), 0 (z > l_o)
    on c_e ("elastin") or c_e and alpha_0 ("both": diagonal collagen deposited at alpha(x), hcmm.Reorienting);
    local=False: uniform insult (1 + f) / 2 -> 1; tether: perivascular push / pull (Weiss Fig. 5, middle row), lateral
    dead forces of tether mN (full tube), alternating sign, on the cross-sections at z = tether_z L for
    tether_day < day <= tether_day + tether_days; insult_day: start of the insult (n_ramp=1: one step);
    noise: elastin heterogeneity, c_e(x) = c_e (1 + noise xi(z, theta)), already in the prestress (homeostatic with the
    field, elastin prestretch adjusted pointwise) and kept on the KO value; xi is one
    fixed field over axial position z (mm) and circumferential angle theta, constant through the thickness, stored in
    elastin_field.npz (generated once: seed 0, Gaussian-smoothed white noise, correlation length 0.5 mm, periodic over
    z in [0, 32) mm and theta in [0, 2 pi), max|xi| = 1); the half tube samples theta in [0, pi];
    orient: deposition angle of the diagonal collagen "prescribed" (alpha_0 insult) or "stress" (emergent, alpha_0 insult
    unused; Hariton 2007 relative to the homeostatic state, stresses as nonlocal means):
        tan alpha_dep = tan alpha_h ((s_tt / s_zz) / (s_tt / s_zz)_h)^gamma,   s_ii = sigma_ii - sigma_rr;
    ends "isometric" (length fixed) or "isotonic" (axial end reactions of the prestressed state held as dead loads),
    end centerlines pinned laterally; wall stress stimulus (common.wall_stress: "trace" tr sigma with the pressure as in
    Weiss, "in-plane" (pressure-free), or
    "von-mises", sign-blind: axial compression from constrained growth feeds back), nonlocal mean over length (default
    sqrt(a h)),
    growth: "fixed" (no part follows F_g, classical CMM: volume from mass, shape from equilibrium and turnover; equal to
    "radial" for this wall, all fibers in the theta-z plane), "isotropic" (fibers lengthen stress-free by J_g^(1/3))
    or "axial" (prescribed lengthening);
    kw: weiss_par values. Output per step: TI = (L1 - L2) / L2 of the centerline; ParaView: biaxial wall stress
    (Fig. 6), volume change J_g, c_e, diagonal fiber angle."""
    log = Log()
    name = name or f"{ends}_{insult}"
    txt, pvd, csv_path, fig = outputs("weiss/tortuosity", *(f"{name}{s}" for s in (".txt", ".pvd", ".csv", ".png")))
    par = weiss_par(**kw)
    a, h, p0, p = float(par["a"]), float(par["h"]), float(par["p_hom"]), float(par["p_gr"])
    length = length or np.sqrt(a * h)
    m = meshlib.half_tube(r_i=a, t=h, L=L, n=n)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"], element="hybrid")
    Q = jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))
    e_r = Q[..., :, 0] if stimulus == "in-plane" else None
    s = np.asarray(fem.gauss_points(m.param, m.conn))
    th, z, l_o = np.pi * s[..., 1], L * s[..., 2], L / 2
    th_hat = np.where(z < l_o, np.pi, 0.0)
    f = 0.5 * (1 - np.cos(2 * np.pi * z / l_o)) * np.exp(-(np.angle(np.exp(1j * (th - th_hat))) / 2) ** 2)
    shape = (1 + f) / 2 if local else np.ones_like(f)
    ramp = lambda k: float(np.clip((ds * k - insult_day) / (ds * n_ramp), 0.0, 1.0))

    top = lambda end: m.nodes[end][np.isclose(m.param[m.nodes[end], 1], 0.5)]
    held = [3 * m.nodes["sym"] + 1, 3 * m.nodes["inlet"] + 2, 3 * top("inlet"), 3 * top("outlet")]
    out_z = 3 * m.nodes["outlet"] + 2
    iso = np.unique(np.concatenate(held + [out_z]))
    bc_iso = lambda p: fem.BC(iso, np.zeros(len(iso)), p=p)

    log(f"Weiss 2020: straight half tube a {a} mm, h {h} mm, L {L} mm, {m.n_elem} hex8 (hybrid); table S2 wall "
        f"({', '.join(q.name for q in WALL_WEISS)}), {stimulus} wall stress stimulus (nonlocal {length:.3f} mm), {growth} growth")
    log(f"insult {'Eq. 8' if local else 'uniform'} on "
        f"{'c_e' if insult == 'elastin' or orient == 'stress' else 'c_e and alpha_0'}, collagen deposition angle "
        f"{orient}{f' (gamma {gamma:g})' if orient == 'stress' else ''} over "
        f"{n_ramp} steps of {ds:g} days, {n_hold} steps held; {ends} ends; insult from day {insult_day:g}; "
        f"elastin noise {noise:g} (elastin_field.npz); "
        f"tether {tether:g} mN at z/L {tuple(tether_z)} on days ({tether_day:g}, {tether_day + tether_days:g}]; "
        f"p {p0 / 1.33322e-4:.0f} -> {p / 1.33322e-4:.0f} mmHg with the insult")
    t0 = time.time()
    field = pathlib.Path(__file__).with_name("elastin_field.npz")
    if not field.exists():
        nx, ny, Lx, a_ref, ell = 1024, 256, 32.0, 0.568, 0.5
        kx, ky = np.meshgrid(2 * np.pi * np.fft.fftfreq(nx, Lx / nx), 2 * np.pi * np.fft.fftfreq(ny, 2 * np.pi * a_ref / ny),
                             indexing="ij")
        g = np.fft.ifft2(np.fft.fft2(np.random.default_rng(0).standard_normal((nx, ny)))
                         * np.exp(-0.5 * (kx ** 2 + ky ** 2) * ell ** 2)).real
        np.savez(field, xi=(g - g.mean()) / np.abs(g - g.mean()).max(), length=Lx)
    grid = np.load(field)
    xi = map_coordinates(grid["xi"], [z / float(grid["length"]) * grid["xi"].shape[0],
                                      th / (2 * np.pi) * grid["xi"].shape[1]], order=1, mode="grid-wrap")
    scale = jnp.asarray(1.0 + noise * xi)
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = Q @ jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])) @ jnp.swapaxes(Q, -1, -2)
    _, u, states = artery_simulate(m, sysm, Q, par, n_steps=0, bc=bc_iso, final=True, log=log, ds=ds,
                                   wall=WALL_WEISS, laws="wall", incompressible=True, growth=growth, G_e=G_e,
                                   fields={"elastin.C10": par["elastin.C10"] * scale} if noise else None)
    log(f"  prestress done in {time.time() - t0:.0f}s")
    bc = bc_iso(p0)
    if ends == "isotonic":
        f_end = sysm.reaction(u, states, bc)
        f_dead = np.zeros(sysm.n_dof)
        f_dead[out_z] = f_end[out_z]
        free = np.unique(np.concatenate(held))
        bc = fem.BC(free, np.zeros(len(free)), f_dead=f_dead, p=p0)
        log(f"  isotonic: axial end force {2 * f_dead.sum() * 1e3:.3f} mN (full tube)")
    f_base = np.zeros(sysm.n_dof) if bc.f_dead is None else bc.f_dead
    f_t = np.zeros(sysm.n_dof)
    for zt, sign in zip(tether_z, (1.0, -1.0) * len(tether_z)):
        zs = m.param[:, 2]
        ring = np.flatnonzero(np.isclose(zs, zs[np.argmin(np.abs(zs - zt))]))
        f_t[3 * ring] += sign * 0.5e-3 * tether / len(ring)
    sw = wall_stress(sysm, u, states, length, e_r, stimulus == "trace")
    states = states.replace(sigma_w=sw, sigma_w_h=sw)

    names = [q.name for q in WALL_WEISS]
    C10 = states.constituents[0].material.C10
    alpha = par["diag+.alpha"]
    e_t, e_z = Q[..., :, 1], Q[..., :, 2]
    series = meshlib.VTKSeries(pvd)
    rows = []
    angle = lambda M: jnp.degrees(jnp.arccos(jnp.abs(jnp.sum(M * e_z, axis=-1))))

    def in_plane(u, states):
        """(s_tt - s_rr, s_zz - s_rr) along the deformed wall directions, nonlocal mean (free of the hybrid pressure)"""
        F, sig = sysm.gauss_F(u), sysm.stress(u, states)
        s = lambda n: jnp.einsum("...i,...ij,...j->...", n, sig, n) / jnp.einsum("...i,...i->...", n, n)
        rr = s(jnp.linalg.solve(jnp.swapaxes(F, -1, -2), Q[..., :, 0][..., None])[..., 0])
        return [sysm.nonlocal_mean(s(jnp.einsum("...ij,...j->...i", F, Q[..., :, i])) - rr, length) for i in (1, 2)]

    if orient == "stress":
        s_tt, s_zz = in_plane(u, states)
        ratio_h = s_tt / s_zz

    def record(k, u, states):
        M = states.constituents[names.index("diag+")].material.M
        M_dep = getattr(states.constituents[names.index("diag+")].production, "M_dep", M)
        e = np.asarray(sysm.gauss_F(u) @ Q)
        e = e / np.linalg.norm(e, axis=-2, keepdims=True)
        biaxial = np.einsum("egia,egij,egja->ega", e, sysm.total_stress(u, states), e)
        series.write(ds * k, m.X, m.conn, *sysm.fields(u, states, dict(
            sigma_tt_kPa=1e3 * biaxial[..., 1], sigma_zz_kPa=1e3 * biaxial[..., 2],
            c_e_rel=states.constituents[0].material.C10 / par["elastin.C10"], insult=jnp.asarray(ramp(k) * shape),
            alpha_diag_deg=angle(M), alpha_dep_deg=angle(M_dep),
            wall_stress_rel=states.sigma_w / states.sigma_w_h, J_g=det3(states.F_g))))

    record(0, u, states)
    t0 = time.time()
    n_steps = round(insult_day / ds) + n_ramp + n_hold
    for k in range(1, n_steps + 1):
        w = jnp.asarray(ramp(k) * shape)
        cs = list(states.constituents)
        cs[0] = cs[0].replace(material=materials.NeoHookeanInc(C10 + w * (KO["C10"] * scale - C10)))
        if orient == "stress":
            s_tt, s_zz = in_plane(u, states)
            al = np.arctan(np.tan(alpha) * (np.maximum(s_tt, 0.0) / np.maximum(s_zz, 1e-12) / ratio_h) ** gamma)
        elif insult == "both":
            al = alpha + w * (KO["alpha"] - alpha)
        if orient == "stress" or insult == "both":
            al = jnp.asarray(al)[..., None]
            for d, sign in (("diag+", 1.0), ("diag-", -1.0)):
                c = cs[names.index(d)]
                cs[names.index(d)] = c.replace(production=c.production.replace(
                    M_dep=jnp.sin(al) * e_t + sign * jnp.cos(al) * e_z))
        states = states.replace(constituents=cs)
        bc = replace(bc, p=p0 + ramp(k) * (p - p0), f_dead=f_base + (f_t if tether_day < ds * k <= tether_day + tether_days else 0.0))
        u, states, _ = sysm.run(states.replace(sigma_w=wall_stress(sysm, u, states, length, e_r, stimulus == "trace")), [bc], u0=u)
        c = centerline(m, u)
        J_g = np.asarray(det3(states.F_g))
        diag = states.constituents[names.index("diag+")]
        al_fib = np.asarray(angle(diag.material.M))
        al_dep = np.asarray(angle(getattr(diag.production, "M_dep", diag.material.M)))
        lam_t = np.linalg.norm(np.einsum("egij,egj->egi", np.asarray(sysm.gauss_F(u)), np.asarray(e_t)), axis=-1)
        rows.append(dict(day=ds * k, ramp=ramp(k), TI=tortuosity(c),
                         L1=np.linalg.norm(np.diff(c, axis=0), axis=1).sum() / L, L2=np.linalg.norm(c[-1] - c[0]) / L,
                         deflection=np.abs(c[:, 0]).max() / a, J_g_mean=J_g.mean(), J_g_max=J_g.max(),
                         circ_mean=lam_t.mean(), circ_max=lam_t.max(), alpha_mean=al_fib.mean(),
                         alpha_min=al_fib.min(), alpha_max=al_fib.max(), alpha_dep_mean=al_dep.mean(),
                         axial_force_mN=2e3 * float((sysm.reaction(u, states, bc)
                                                     + (0.0 if bc.f_dead is None else bc.f_dead))[out_z].sum())))
        if k % record_every == 0 or k == n_steps:
            record(k, u, states)
        if k % max(n_steps // 10, 1) == 0:
            r = rows[-1]
            log(f"  day {r['day']:5.0f}  ramp {r['ramp']:.2f}  TI {r['TI']:.2e}  L1/L {r['L1']:.4f}  L2/L {r['L2']:.4f}"
                f"  deflection/a {r['deflection']:.3f}  J_g mean {r['J_g_mean']:.4f} max {r['J_g_max']:.4f}"
                f"  circ. stretch mean {r['circ_mean']:.3f} max {r['circ_max']:.3f}  axial force {r['axial_force_mN']:.2f} mN"
                f"  alpha mean {r['alpha_mean']:.2f} ({r['alpha_min']:.2f}-{r['alpha_max']:.2f}) dep {r['alpha_dep_mean']:.2f} deg"
                f"  ({time.time() - t0:.0f}s)")
    log(f"  ParaView: {pvd.name} ({len(series.entries)} time steps)")
    write_csv(rows, csv_path)
    col = lambda key: np.array([r[key] for r in rows])
    t = col("ramp") if n_hold == 0 else col("day")
    plotting.panels([dict(t=t, series={"TI": col("TI")}, title="Tortuosity index", ylabel="(-)"),
                     dict(t=t, series={"L1 / L": col("L1"), "L2 / L": col("L2")}, title="Centerline length",
                          ylabel="(-)"),
                     dict(t=t, series={"max |x_c| / a": col("deflection")}, title="Lateral deflection", ylabel="(-)"),
                     dict(t=t, series={"mean": col("J_g_mean"), "max": col("J_g_max")}, title="Volume growth J_g",
                          ylabel="(-)"),
                     dict(t=t, series={"mean": col("circ_mean"), "max": col("circ_max")},
                          title="Circumferential stretch (diameter)", ylabel="(-)")], fig)
    log.save(txt)
