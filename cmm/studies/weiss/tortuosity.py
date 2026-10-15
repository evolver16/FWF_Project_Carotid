"""Weiss et al. 2020, Fig. 5/6: tortuosity of a straight aorta from a local loss of elastic fiber integrity (Eq. 8),
isometric vs isotonic ends."""

import time
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

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


def weiss_tortuosity(ends="isometric", insult="both", n=(2, 16, 120), L=14.0, n_ramp=200, n_hold=0, ds=1.0,
                     growth="isotropic", stimulus="trace", length=None, record_every=10, name=None, **kw):
    """Straight half tube (in vivo a, h, length L = 2 l_o ~ 12 diameters as in Weiss Fig. 5/6, hybrid hex8), wall of
    table S2 prestressed at 120 mmHg with
    fixed ends (elastin from G_e = diag(1/(G_theta G_z), G_theta, G_z), stress-free state fixed); insult ramped over
    n_ramp steps of ds days (slow against T: quasi-static, compared over dv / dv_KO as Weiss' equilibrated states),
    then held n_hold steps (Eq. 8); pressure ramped with the insult from p_hom to p_gr (kw, MPa; default both 120 mmHg):
        v(x) = v_WT + r (v_KO - v_WT) (1 + f(x)) / 2,   r = min(k / n_ramp, 1)
        f    = 1/2 (1 - cos(2 pi z / l_o)) exp(-((theta - theta_hat) / 2)^2),   theta_hat = pi (z < l_o), 0 (z > l_o)
    on c_e ("elastin") or c_e and alpha_0 ("both": diagonal collagen deposited at alpha(x), hcmm.Reorienting);
    ends "isometric" (length fixed) or "isotonic" (axial end reactions of the prestressed state held as dead loads),
    end centerlines pinned laterally; wall stress stimulus (common.wall_stress: "trace" tr sigma with the pressure as in
    Weiss, "in-plane" (pressure-free), or
    "von-mises", sign-blind: axial compression from constrained growth feeds back), nonlocal mean over length (default
    sqrt(a h)),
    growth: mixture F_g "isotropic" (no preferred direction; with elastin fixed and the trace stimulus the direction
    follows from mechanics), "radial" (thickening) or "axial" (prescribed lengthening);
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

    top = lambda end: m.nodes[end][np.isclose(m.param[m.nodes[end], 1], 0.5)]
    held = [3 * m.nodes["sym"] + 1, 3 * m.nodes["inlet"] + 2, 3 * top("inlet"), 3 * top("outlet")]
    out_z = 3 * m.nodes["outlet"] + 2
    iso = np.unique(np.concatenate(held + [out_z]))
    bc_iso = lambda p: fem.BC(iso, np.zeros(len(iso)), p=p)

    log(f"Weiss 2020: straight half tube a {a} mm, h {h} mm, L {L} mm, {m.n_elem} hex8 (hybrid); table S2 wall "
        f"({', '.join(q.name for q in WALL_WEISS)}), {stimulus} wall stress stimulus (nonlocal {length:.3f} mm), {growth} growth")
    log(f"insult Eq. 8 on {'c_e' if insult == 'elastin' else 'c_e and alpha_0'} over {n_ramp} steps of {ds:g} days, "
        f"{n_hold} steps held; {ends} ends; p {p0 / 1.33322e-4:.0f} -> {p / 1.33322e-4:.0f} mmHg with the insult")
    t0 = time.time()
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = Q @ jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])) @ jnp.swapaxes(Q, -1, -2)
    _, u, states = artery_simulate(m, sysm, Q, par, n_steps=0, bc=bc_iso, final=True, log=log, ds=ds,
                                   wall=WALL_WEISS, laws="wall", incompressible=True, growth=growth, G_e=G_e)
    log(f"  prestress done in {time.time() - t0:.0f}s")
    bc = bc_iso(p0)
    if ends == "isotonic":
        f_end = sysm.reaction(u, states, bc)
        f_dead = np.zeros(sysm.n_dof)
        f_dead[out_z] = f_end[out_z]
        free = np.unique(np.concatenate(held))
        bc = fem.BC(free, np.zeros(len(free)), f_dead=f_dead, p=p0)
        log(f"  isotonic: axial end force {2 * f_dead.sum() * 1e3:.3f} mN (full tube)")
    sw = wall_stress(sysm, u, states, length, e_r, stimulus == "trace")
    states = states.replace(sigma_w=sw, sigma_w_h=sw)

    names = [q.name for q in WALL_WEISS]
    C10 = states.constituents[0].material.C10
    alpha = par["diag+.alpha"]
    e_t, e_z = Q[..., :, 1], Q[..., :, 2]
    series = meshlib.VTKSeries(pvd)
    rows = []

    def record(k, u, states):
        M = states.constituents[names.index("diag+")].material.M
        e = np.asarray(sysm.gauss_F(u) @ Q)
        e = e / np.linalg.norm(e, axis=-2, keepdims=True)
        biaxial = np.einsum("egia,egij,egja->ega", e, sysm.total_stress(u, states), e)
        series.write(ds * k, m.X, m.conn, *sysm.fields(u, states, dict(
            sigma_tt_kPa=1e3 * biaxial[..., 1], sigma_zz_kPa=1e3 * biaxial[..., 2],
            c_e_rel=states.constituents[0].material.C10 / C10, insult=jnp.asarray(f),
            alpha_diag_deg=jnp.degrees(jnp.arccos(jnp.abs(jnp.sum(M * e_z, axis=-1)))),
            wall_stress_rel=states.sigma_w / states.sigma_w_h, J_g=det3(states.F_g))))

    record(0, u, states)
    t0 = time.time()
    n_steps = n_ramp + n_hold
    for k in range(1, n_steps + 1):
        w = jnp.asarray(min(k / n_ramp, 1.0) * (1 + f) / 2)
        cs = list(states.constituents)
        cs[0] = cs[0].replace(material=materials.NeoHookeanInc(C10 + w * (KO["C10"] - C10)))
        if insult == "both":
            al = (alpha + w * (KO["alpha"] - alpha))[..., None]
            for d, sign in (("diag+", 1.0), ("diag-", -1.0)):
                c = cs[names.index(d)]
                cs[names.index(d)] = c.replace(production=c.production.replace(
                    M_dep=jnp.sin(al) * e_t + sign * jnp.cos(al) * e_z))
        states = states.replace(constituents=cs)
        bc = replace(bc, p=p0 + min(k / n_ramp, 1.0) * (p - p0))
        u, states, _ = sysm.run(states.replace(sigma_w=wall_stress(sysm, u, states, length, e_r, stimulus == "trace")), [bc], u0=u)
        c = centerline(m, u)
        J_g = np.asarray(det3(states.F_g))
        lam_t = np.linalg.norm(np.einsum("egij,egj->egi", np.asarray(sysm.gauss_F(u)), np.asarray(e_t)), axis=-1)
        rows.append(dict(day=ds * k, ramp=min(k / n_ramp, 1.0), TI=tortuosity(c),
                         L1=np.linalg.norm(np.diff(c, axis=0), axis=1).sum() / L, L2=np.linalg.norm(c[-1] - c[0]) / L,
                         deflection=np.abs(c[:, 0]).max() / a, J_g_mean=J_g.mean(), J_g_max=J_g.max(),
                         circ_mean=lam_t.mean(), circ_max=lam_t.max(),
                         axial_force_mN=2e3 * float((sysm.reaction(u, states, bc)
                                                     + (0.0 if bc.f_dead is None else bc.f_dead))[out_z].sum())))
        if k % record_every == 0 or k == n_steps:
            record(k, u, states)
        if k % max(n_steps // 10, 1) == 0:
            r = rows[-1]
            log(f"  day {r['day']:5.0f}  ramp {r['ramp']:.2f}  TI {r['TI']:.2e}  L1/L {r['L1']:.4f}  L2/L {r['L2']:.4f}"
                f"  deflection/a {r['deflection']:.3f}  J_g mean {r['J_g_mean']:.4f} max {r['J_g_max']:.4f}"
                f"  circ. stretch mean {r['circ_mean']:.3f} max {r['circ_max']:.3f}  axial force {r['axial_force_mN']:.2f} mN"
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
