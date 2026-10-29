"""Weiss 2020 tortuosity on a 3D rod (rod.Rod): straight tube, centerline free to bend in any direction, each section
a thin ring of HCMM material points; reduced model of weiss_tortuosity (full=True)."""

import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates

import hcmm
import materials
from analysis import tortuosity
from fem import mesh as meshlib
from plotting import plt
from rod import Rod
from studies.common import Log, outputs, write_csv
from studies.weiss.model import KO, WALL_WEISS, weiss_par
from wall import assemble


def weiss_rod(insult_day=10.0, n_ramp=1, n_hold=100, ds=2.0, L=14.0, n_el=60, n_ring=16, orient="stress", gamma=1.0,
              local=False, noise=0.01, length=None, record_every=5, name="rod", **kw):
    """Clamped isometric ends, Weiss wall (table S2), fixed growth, prestress as in the FE; elastin c_e -> KO from
    insult_day over n_ramp steps of ds days (local: Eq. 8 shape), n_hold steps held; stimulus tr sigma, nonlocal over
    length (default sqrt(a h)); diagonal collagen deposited at the emergent angle (orient "stress", gamma) or
    prescribed; symmetry breaking: c_e (1 + noise xi(z, theta)) from elastin_field.npz over the whole circumference
    (as the FE full tube). Output per step: TI, deflection, bending direction, J_g, alpha; ParaView: deformed wall."""
    log = Log()
    txt, csv_path, fig, pvd = outputs("weiss/rod", *(f"{name}{s}" for s in (".txt", ".csv", ".png", ".pvd")))
    par = weiss_par(**kw)
    a0, H, p = float(par["a"]), float(par["h"]), float(par["p_hom"])
    length = length or np.sqrt(a0 * H)
    X = np.stack([np.zeros(n_el + 1), np.zeros(n_el + 1), np.linspace(0.0, L, n_el + 1)], -1)
    conn = np.stack([np.arange(n_el), np.arange(1, n_el + 1)], -1)
    rod = Rod(X, conn, np.zeros(n_el, int), np.full((n_el, 2), a0), np.full((n_el, 3), H), hcmm, clamped=(0, n_el),
              n_ring=n_ring)
    shp = (3 * n_el * n_ring, 1)
    zz = rod.X_points[:, 2].reshape(shp)
    tt = np.mod(np.arctan2(rod.X_points[:, 1], rod.X_points[:, 0]), 2 * np.pi).reshape(shp)
    grid = np.load(pathlib.Path(__file__).with_name("elastin_field.npz"))
    xi = map_coordinates(grid["xi"], [zz / float(grid["length"]) * grid["xi"].shape[0],
                                      tt / (2 * np.pi) * grid["xi"].shape[1]], order=1, mode="grid-wrap")
    scale = jnp.asarray(1.0 + noise * xi)
    l_o = L / 2
    th_hat = np.where(zz < l_o, np.pi, 0.0)
    f = 0.5 * (1 - np.cos(2 * np.pi * zz / l_o)) * np.exp(-(np.angle(np.exp(1j * (tt - th_hat))) / 2) ** 2)
    shape = jnp.asarray((1 + f) / 2 if local else np.ones_like(f))
    ramp = lambda k: float(np.clip((ds * k - insult_day) / (ds * n_ramp), 0.0, 1.0))

    log(f"Weiss 2020 on a 3D rod: a {a0} mm, h {H} mm, L {L} mm, {n_el} Hermite elements x 3 sections x {n_ring} "
        f"ring points ({len(rod.free)} DOFs); clamped isometric ends; tr sigma stimulus (nonlocal {length:.3f} mm)")
    log(f"insult {'Eq. 8' if local else 'uniform'} on c_e, deposition angle {orient}"
        f"{f' (gamma {gamma:g})' if orient == 'stress' else ''} over {n_ramp} steps of {ds:g} days, {n_hold} held; "
        f"insult from day {insult_day:g}; elastin noise {noise:g} (full circumference)")
    t0 = time.time()
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = jnp.broadcast_to(jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])), shp + (3, 3))
    Q = jnp.broadcast_to(jnp.eye(3), G_e.shape)
    states, it, du = rod.prestress(lambda G: assemble(G, Q, par, WALL_WEISS, ds=ds, laws="wall", incompressible=True,
                                                      growth="fixed", fields={"elastin.C10": par["elastin.C10"] * scale}),
                                   G_e, p)
    log(f"  prestress: {it} iterations, max|q - q_0| {du:.1e}  ({time.time() - t0:.0f}s)")
    q = rod.q0
    s_tt, s_zz = rod.stresses(q, states)
    sw = rod.nonlocal_mean(s_tt + s_zz, length)
    states = states.replace(sigma_w=sw, sigma_w_h=sw)
    ratio_h = rod.nonlocal_mean(s_tt, length) / rod.nonlocal_mean(s_zz, length)
    names = [c.name for c in WALL_WEISS]
    C10 = states.constituents[0].material.C10
    alpha = par["diag+.alpha"]
    e_t, e_z = jnp.array([0.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    angle = lambda M: np.degrees(np.arccos(np.abs(np.asarray(M @ e_z))))
    J_g = lambda st: np.asarray(jax.vmap(jax.vmap(hcmm.J_target))(st))
    series = meshlib.VTKSeries(pvd)
    X_ref = rod.wall(q, states)

    def record(day, q, st):
        X, conn = rod.wall(q, st)
        s_tt, s_zz = (np.asarray(v) for v in rod.stresses(q, st))
        F = np.asarray(rod.F(q, st))
        diag = st.constituents[names.index("diag+")]
        per = lambda v: np.repeat(np.asarray(v).reshape(-1), 2)
        fields = dict(J_g=per(J_g(st)), sigma_tt_kPa=per(1e3 * s_tt), sigma_zz_kPa=per(1e3 * s_zz),
                      lambda_theta=per(F[..., 1, 1]), lambda_z=per(F[..., 2, 2]),
                      c_e_rel=per(st.constituents[0].material.C10 / par["elastin.C10"]),
                      alpha_diag_deg=per(angle(diag.material.M)), alpha_dep_deg=per(angle(diag.production.M_dep)),
                      wall_stress_rel=per(st.sigma_w / st.sigma_w_h))
        series.write(day, X_ref[0], conn, fields | {"u": X - X_ref[0]}, {k: v[conn].mean(1) for k, v in fields.items()})

    rows, shapes = [], {}
    n_steps = round(insult_day / ds) + n_ramp + n_hold
    record(0.0, q, states)
    t0 = time.time()
    for k in range(1, n_steps + 1):
        cs = list(states.constituents)
        cs[0] = cs[0].replace(material=materials.NeoHookeanInc(C10 + ramp(k) * shape * (KO["C10"] * scale - C10)))
        s_tt, s_zz = rod.stresses(q, states)
        if orient == "stress":
            r_tt, r_zz = rod.nonlocal_mean(s_tt, length), rod.nonlocal_mean(s_zz, length)
            al = jnp.arctan(jnp.tan(alpha) * (jnp.maximum(r_tt, 0.0) / jnp.maximum(r_zz, 1e-12) / ratio_h) ** gamma)
            for d, sign in (("diag+", 1.0), ("diag-", -1.0)):
                c = cs[names.index(d)]
                cs[names.index(d)] = c.replace(production=c.production.replace(
                    M_dep=jnp.sin(al)[..., None] * e_t + sign * jnp.cos(al)[..., None] * e_z))
        states = states.replace(constituents=cs, sigma_w=rod.nonlocal_mean(s_tt + s_zz, length))
        q, it = rod.solve(q, states, p)
        states = rod.commit(q, states)
        c = rod.positions(q)
        off = c[:, :2]
        i = np.argmax(np.linalg.norm(off, axis=1))
        jg, al_f = J_g(states), angle(states.constituents[names.index("diag+")].material.M)
        rows.append(dict(day=ds * k, ramp=ramp(k), TI=tortuosity(c),
                         L1=np.linalg.norm(np.diff(c, axis=0), axis=1).sum() / L,
                         deflection=np.linalg.norm(off[i]) / a0, direction_deg=np.degrees(np.arctan2(off[i, 1], off[i, 0])),
                         out_of_plane=np.abs(off @ np.array([-off[i, 1], off[i, 0]]) / (np.linalg.norm(off[i]) or 1)).max() / a0,
                         J_g_mean=jg.mean(), J_g_max=jg.max(), alpha_mean=al_f.mean(), alpha_min=al_f.min(),
                         alpha_max=al_f.max()))
        if k % record_every == 0 or k == n_steps:
            shapes[ds * k] = c
            record(ds * k, q, states)
            r = rows[-1]
            log(f"  day {ds * k:5g}  ramp {r['ramp']:.2f}  TI {r['TI']:.2e}  deflection/a {r['deflection']:.3f} "
                f"towards {r['direction_deg']:.0f} deg (out of plane/a {r['out_of_plane']:.3f})  J_g mean "
                f"{r['J_g_mean']:.4f} max {r['J_g_max']:.4f}  alpha mean {r['alpha_mean']:.2f} ({r['alpha_min']:.2f}-"
                f"{r['alpha_max']:.2f}) deg  (Newton {it}, {time.time() - t0:.0f}s)")

    write_csv(rows, csv_path)
    fg, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot([r["day"] for r in rows], [r["TI"] for r in rows])
    ax1.set(xlabel="day", ylabel="TI", yscale="log", title="tortuosity index")
    for day in list(shapes)[::max(1, len(shapes) // 6)] + [list(shapes)[-1]]:
        c = shapes[day]
        ax2.plot(c[:, 2], np.linalg.norm(c[:, :2], axis=1), label=f"day {day:g}")
    ax2.set(xlabel="z [mm]", ylabel="lateral deflection [mm]", title="centerline")
    ax2.legend(fontsize=8)
    fg.tight_layout()
    fg.savefig(fig, dpi=150)
    log(f"  figure {fig.name}, table {csv_path.name}, ParaView {pvd.name} ({len(series.entries)} time steps)")
    log.save(txt)
