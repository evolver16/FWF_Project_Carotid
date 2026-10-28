"""Hariton et al. 2007 on the straight Weiss tube: diagonal collagen angle from the stresses, homeostatic and after
elastin loss."""

import time

import jax.numpy as jnp
import numpy as np

import fem
import hcmm
import materials
import plotting
from fem import mesh as meshlib
from fem.tensor3 import det3
from studies.common import Log, hariton_angle, outputs, wall_stress, write_csv
from studies.model_e import artery_simulate
from studies.weiss.model import KO, WALL_WEISS, weiss_par
from verification.fem_tube import cylinder_bc


def free_angle(free=True, n=(8, 30, 1), etype="hex8", angle_iter=40, angle_tol=1e-3, relax=1.0, n_ramp=200, n_hold=300,
               ds=1.0, length=None, tol=1e-10, name="free_angle", **kw):
    """Straight tube, table S2 wall at 120 mmHg (quarter cylinder, plane strain, isometric, hybrid hex8 Q1/P0 or tet10
    P2/P1, n = (radial, circumferential, axial)); diagonal collagen angle from the axis set by the stress, Hariton 2007
    Eqs. 7, 8, 11 with the principal directions theta, z of the axisymmetric tube:
        tan alpha = sigma_tt / sigma_zz   (total Cauchy stress, per Gauss point)
    homeostasis: prestress until u = 0, alpha <- alpha + relax (rule - alpha), repeated from alpha_0 of table S2 until
    max|d alpha| < angle_tol deg; then c_e -> KO over n_ramp steps of ds days, held n_hold steps, diagonal collagen
    deposited at the rule's angle (hcmm.Reorienting); tr sigma stimulus, nonlocal over length (default sqrt(a h), over
    the whole wall; pointwise it diverges with Q1/P0); free=False: alpha_0 of table S2 kept, no
    reorientation (reference). kw: weiss_par values (e.g. h for a thicker wall)"""
    log = Log()
    txt, csv_path, fig = outputs("weiss/free_angle", f"{name}.txt", f"{name}.csv", f"{name}.png")
    par = weiss_par(**kw)
    a, h, p0 = float(par["a"]), float(par["h"]), float(par["p_hom"])
    length = length or np.sqrt(a * h)
    m = meshlib.quarter_cylinder(r_i=a, t=h, L=h, n=n, etype=etype)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces[(0, -1)], element="hybrid" if etype == "hex8" else "hybrid_p1")
    xg = np.asarray(fem.gauss_points(m.X, m.conn))
    Q = jnp.asarray(meshlib.cylinder_basis(xg))
    e_t, e_z = Q[..., :, 1], Q[..., :, 2]
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = Q @ jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])) @ jnp.swapaxes(Q, -1, -2)
    names = [q.name for q in WALL_WEISS]
    r = np.linalg.norm(xg[..., :2], axis=-1)
    layers = (np.arange(2 * n[0]) + 0.5) / (2 * n[0])
    layer = np.clip(((r - a) / h * 2 * n[0]).astype(int), 0, 2 * n[0] - 1)
    profile = lambda v: np.bincount(layer.ravel(), np.asarray(v).ravel()) / np.bincount(layer.ravel())
    angle = lambda M: np.degrees(np.arccos(np.abs(np.asarray(jnp.sum(M * e_z, axis=-1)))))
    log(f"Hariton 2007 free collagen angle, straight tube a {a} mm, h {h} mm (h/a {h / a:.3f}), mesh {n} = "
        f"{m.n_elem} {etype} ({sysm.element}), {len(layers)} radial bins through the wall; table S2 wall at "
        f"{p0 / 1.33322e-4:.0f} mmHg, isometric; elastin c_e -> KO over {n_ramp} x {ds:g} d, held {n_hold}; "
        f"tr sigma stimulus, nonlocal {length:.4f} mm")

    rule = lambda u, states: hariton_angle(sysm, Q, u, states)
    al =np.full(r.shape, float(par["diag+.alpha"]))
    t0 = time.time()
    for it in range(1, angle_iter + 1):
        _, u, states = artery_simulate(m, sysm, Q, par, n_steps=0, bc=lambda p: cylinder_bc(m, p), final=True, ds=ds,
                                       wall=WALL_WEISS, laws="wall", incompressible=True, growth="fixed", G_e=G_e,
                                       fields={"diag+.alpha": al, "diag-.alpha": al}, anderson=2, pre_iter=200)
        target, s_tt, s_zz = rule(u, states)
        d = np.degrees(np.abs(target - al)).max()
        log(f"  angle iteration {it:2d}  alpha {np.degrees(al).min():.2f}-{np.degrees(al).max():.2f} deg  "
            f"rule {np.degrees(target).min():.2f}-{np.degrees(target).max():.2f}  max change {d:.2e} deg  "
            f"s_tt / s_zz {1e3 * s_tt.mean():.1f} / {1e3 * s_zz.mean():.1f} kPa  ({time.time() - t0:.0f}s)")
        if d < angle_tol or not free:
            break
        al = al + relax * (target - al)
    al_h = profile(np.degrees(al))
    log("  homeostatic profile, (r - a) / h: alpha (deg), s_tt, s_zz (kPa)")
    for x, v, st, sz in zip(layers, al_h, profile(1e3 * s_tt), profile(1e3 * s_zz)):
        log(f"    {x:.3f}  {v:6.2f}  {st:7.1f}  {sz:7.1f}")
    log(f"  Weiss WT alpha_0 {np.degrees(float(par['diag+.alpha'])):.1f} deg (table S2)")

    states = states.replace(sigma_w=(sw := wall_stress(sysm, u, states, length, None, True)), sigma_w_h=sw)
    C10 = states.constituents[0].material.C10
    bc = cylinder_bc(m, p0)
    rows, profiles = [], {"day 0": angle(states.constituents[names.index("diag+")].material.M)}
    t0 = time.time()
    for k in range(1, n_ramp + n_hold + 1):
        w = min(k / n_ramp, 1.0)
        cs = list(states.constituents)
        cs[0] = cs[0].replace(material=materials.NeoHookeanInc(C10 + w * (KO["C10"] - C10)))
        al = jnp.asarray(rule(u, states)[0] if free else np.radians(angle(cs[names.index("diag+")].material.M)))[..., None]
        for dname, sign in (("diag+", 1.0), ("diag-", -1.0)):
            c = cs[names.index(dname)]
            cs[names.index(dname)] = c.replace(production=c.production.replace(
                M_dep=jnp.sin(al) * e_t + sign * jnp.cos(al) * e_z))
        states = states.replace(constituents=cs, sigma_w=wall_stress(sysm, u, states, length, None, True))
        u, states, _ = sysm.run(states, [bc], u0=u, tol=tol)
        diag = states.constituents[names.index("diag+")]
        al_fib = profile(angle(diag.material.M))
        _, s_tt, s_zz = rule(u, states)
        lam_t = np.linalg.norm(np.einsum("egij,egj->egi", np.asarray(sysm.gauss_F(u)), np.asarray(e_t)), axis=-1)
        rows.append(dict(day=ds * k, ramp=w, alpha_inner=al_fib[0], alpha_mean=al_fib.mean(), alpha_outer=al_fib[-1],
                         alpha_dep_mean=float(np.degrees(np.asarray(al)).mean()), s_tt_kPa=1e3 * s_tt.mean(),
                         s_zz_kPa=1e3 * s_zz.mean(), lam_t_inner=profile(lam_t)[0],
                         J_g_mean=float(np.asarray(det3(states.F_g)).mean())))
        if k == n_ramp:
            profiles[f"day {ds * k:g} (KO reached)"] = angle(diag.material.M)
        if k % max((n_ramp + n_hold) // 50, 1) == 0:
            q = rows[-1]
            log(f"  day {q['day']:5.0f}  ramp {q['ramp']:.2f}  alpha inner / mean / outer {q['alpha_inner']:.2f} / "
                f"{q['alpha_mean']:.2f} / {q['alpha_outer']:.2f} deg (dep {q['alpha_dep_mean']:.2f})  "
                f"s_tt / s_zz {q['s_tt_kPa']:.1f} / {q['s_zz_kPa']:.1f} kPa  lam_t inner {q['lam_t_inner']:.4f}  "
                f"J_g {q['J_g_mean']:.4f}  ({time.time() - t0:.0f}s)")
    profiles[f"day {ds * (n_ramp + n_hold):g}"] = angle(states.constituents[names.index("diag+")].material.M)
    log("  fiber angle through the wall, (r - a) / h: " + "  ".join(profiles))
    for i, x in enumerate(layers):
        log(f"    {x:.3f}  " + "  ".join(f"{profile(v)[i]:6.2f}" for v in profiles.values()))
    log(f"  Weiss KO alpha_0 {np.degrees(KO['alpha']):.1f} deg (Fbln5-/-)")
    write_csv(rows, csv_path)
    col = lambda key: np.array([q[key] for q in rows])
    plotting.panels([
        dict(t=layers, series={k: profile(v) for k, v in profiles.items()}
             | {"Weiss WT": np.full(len(layers), np.degrees(float(par["diag+.alpha"]))),
                "Weiss KO": np.full(len(layers), np.degrees(KO["alpha"]))},
             title="Diagonal collagen angle through the wall", ylabel="alpha from the axis (deg)",
             xlabel="(r - a) / h (reference)"),
        dict(t=col("day"), series={"inner": col("alpha_inner"), "mean": col("alpha_mean"),
                                   "outer": col("alpha_outer"), "deposition (mean)": col("alpha_dep_mean")},
             title="Diagonal collagen angle", ylabel="alpha from the axis (deg)"),
        dict(t=col("day"), series={"sigma_tt": col("s_tt_kPa"), "sigma_zz": col("s_zz_kPa")},
             title="Mean wall stress", ylabel="(kPa)"),
        dict(t=col("day"), series={"inner circ. stretch": col("lam_t_inner"), "J_g mean": col("J_g_mean")},
             title="Inner circumferential stretch, volume", ylabel="(-)")], fig)
    log.save(txt)
