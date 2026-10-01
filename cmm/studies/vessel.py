"""General geometries: bent stenotic vessel from an Abaqus .inp mesh with a Laplace wall basis, and mass
redistribution in a straight half tube (end rotation, local set-point change)."""

import time

import jax
import jax.numpy as jnp
import numpy as np

import fem
import hcmm
import plotting
from fem import mesh as meshlib
from fem.elements import face_for
from fem.tensor3 import det3
from studies.artery import artery_par, artery_simulate
from studies.common import Log, outputs, surface_area, write_csv


def dof_bc(m, p, fixed):
    """u_comp = 0 on each (node set, comp) in fixed, pressure p on the pressure faces"""
    dofs = np.unique(np.concatenate([3 * m.nodes[name] + comp for name, comp in fixed]))
    return fem.BC(dofs, np.zeros(len(dofs)), p=p)


def ring_radius(m, u, ring):
    """Mean distance of a half-ring of nodes to the midpoint of its two ends (deformed)"""
    x = jnp.asarray(m.X)[ring] + jnp.asarray(u).reshape(-1, 3)[ring]
    return jnp.mean(jnp.linalg.norm(x - 0.5 * (x[0] + x[-1]), axis=1))


# ---------- bent stenotic vessel: .inp round trip, Laplace wall basis, G&R, adjoint ----------

def stenotic_vessel(etype="hex8", n=None, n_steps=100, check_n=((2, 15, 1), (4, 30, 1), (8, 60, 1))):
    n = n or {"hex8": (3, 24, 40), "tet10": (1, 12, 20)}[etype]
    log = Log()
    tag = "" if etype == "hex8" else f"_{etype}"
    txt, inp, pvd, csv_path, fig = outputs("vessel/stenotic_vessel", *(f"vessel{tag}{s}" for s in (
        ".txt", ".inp", ".pvd", ".csv", ".png")))

    log(f"1) Abaqus .inp round trip (bent stenotic half tube, {etype})")
    src = meshlib.bent_tube(n=n, etype=etype)
    meshlib.write_inp(inp, src)
    m = meshlib.read_inp(inp)

    def same_faces(a, b):
        """same faces with the same outward normals"""
        ia, ib = np.lexsort(np.sort(a, 1).T), np.lexsort(np.sort(b, 1).T)
        a, b = a[ia], b[ib]
        c = face_for(a.shape[1]).n_corners - 1
        normal = lambda q: np.cross(src.X[q[:, 1]] - src.X[q[:, 0]], src.X[q[:, c]] - src.X[q[:, 0]])
        return np.array_equal(np.sort(a, 1), np.sort(b, 1)) and bool(np.all(np.sum(normal(a) * normal(b), 1) > 0))

    names = ("inner", "outer", "inlet", "outlet")
    ok = (np.abs(m.X - src.X).max() < 1e-9 and np.array_equal(m.conn, src.conn)
          and all(same_faces(src.faces[k], m.faces[k]) for k in names)
          and all(np.array_equal(src.nodes[k], m.nodes[k]) for k in names + ("sym",)))
    log(f"  {m.n_elem} {etype}, {m.n_nodes} nodes, sets {sorted(m.nodes)}, surfaces {sorted(m.faces)}  "
        f"{'PASS' if ok else 'FAIL'}")
    _, _, bq = meshlib.boundary_faces(m)
    per_quad = 1 if etype == "hex8" else 2
    log(f"  boundary faces {len(bq)} = inner + outer + inlet + outlet + sym "
        f"{sum(len(m.faces[k]) for k in ('inner', 'outer', 'inlet', 'outlet')) + 2 * n[0] * n[2] * per_quad}")

    log("\n2) Laplace wall basis vs analytic cylinder basis at the Gauss points, quarter cylinder")
    for cn in check_n:
        c = meshlib.quarter_cylinder(n=cn)
        Q_exact = meshlib.cylinder_basis(fem.gauss_points(c.X, c.conn))
        Q_lap = fem.wall_basis(c.X, c.conn, c.nodes[(0, -1)], c.nodes[(0, 1)], [c.nodes[(2, -1)]], [c.nodes[(2, 1)]])
        ang = np.degrees(np.arccos(np.clip(np.sum(Q_exact * Q_lap, axis=-2), -1, 1))).max(axis=(0, 1))
        log(f"  n={cn}  max angle e_r {ang[0]:.2e} deg  e_theta {ang[1]:.2e} deg  e_z {ang[2]:.2e} deg")
    sysc = fem.System(c, hcmm, pressure_faces=c.faces[(0, -1)])
    lam = [artery_simulate(c, sysc, jnp.asarray(Q), artery_par(), n_steps=n_steps)[0][-1] for Q in (Q_exact, Q_lap)]
    log(f"  n={cn}  G&R lambda_theta(day {10 * n_steps}): analytic basis {float(lam[0]):.8f}  "
        f"Laplace basis {float(lam[1]):.8f}")

    log(f"\n3) G&R on the bent stenotic vessel, {n_steps} steps")
    Q = fem.wall_basis(m.X, m.conn, m.nodes["inner"], m.nodes["outer"], [m.nodes["inlet"]], [m.nodes["outlet"]])
    orth = np.abs(np.swapaxes(Q, -1, -2) @ Q - np.eye(3)).max()
    log(f"  basis orthonormality {orth:.1e}, det min {np.linalg.det(Q).min():.6f}")
    Qj = jnp.asarray(Q)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"])
    log(f"  {m.n_elem} {etype}, {sysm.n_dof} dofs")
    bc = lambda p: dof_bc(m, p, [("sym", 1), ("inlet", 2), ("outlet", 0)])

    def ring(s):
        """inner nodes at the axial grid line closest to s, ordered in theta"""
        inner = src.nodes["inner"]
        s2 = src.param[inner, 2]
        on = inner[np.isclose(s2, s2[np.argmin(np.abs(s2 - s))])]
        return on[np.argsort(src.param[on, 1])]

    rings = {"throat": ring(0.5), "inlet": ring(0.1)}
    r0 = {k: float(ring_radius(m, np.zeros(sysm.n_dof), v)) for k, v in rings.items()}
    A0 = float(surface_area(m, np.zeros(sysm.n_dof), m.faces["inner"]))
    V0 = float(jnp.sum(sysm.wdet))
    measure = lambda u: (ring_radius(m, u, rings["throat"]) / r0["throat"],
                         ring_radius(m, u, rings["inlet"]) / r0["inlet"],
                         surface_area(m, u, m.faces["inner"]) / A0,
                         jnp.sum(sysm.wdet * det3(sysm.gauss_F(u))) / V0)
    par = artery_par()
    series = meshlib.VTKSeries(pvd)
    basis = {f"e_{k}": Q[..., i] for i, k in enumerate(("r", "theta", "z"))}

    def record(k, u, states):
        F = sysm.gauss_F(u)
        a = F @ Qj[..., 1:2]
        a = a / jnp.linalg.norm(a, axis=-2, keepdims=True)
        s_tt = (jnp.swapaxes(a, -1, -2) @ sysm.stress(u, states) @ a)[..., 0, 0]
        series.write(10.0 * k, m.X, m.conn, *sysm.fields(u, states, dict(sigma_tt_kPa=1e3 * s_tt, **basis)))

    t0 = time.time()
    (thr, inl, area, vol), u, states = artery_simulate(m, sysm, Qj, par, n_steps=n_steps, bc=bc, measure=measure,
                                                       log=log, final=True, record=record)
    log(f"  ParaView: {pvd.name} ({len(series.entries)} time steps, day 0 = prestressed at p_hom)")
    log(f"  done in {time.time() - t0:.1f}s   (LU factorizations {sysm.linear.factorizations})")
    for k in (0, 9, 49, 99):
        if k < n_steps:
            log(f"  day {10 * (k + 1):4d}  lumen radius stretch throat {float(thr[k]):.4f}  inlet {float(inl[k]):.4f}"
                f"   lumen area {float(area[k]):.4f}   wall volume {float(vol[k]):.4f}")

    log("\n4) adjoint gradient of the throat radius stretch at the end")
    t0 = time.time()
    g = jax.grad(lambda p: artery_simulate(m, sysm, Qj, p, n_steps=n_steps, bc=bc, measure=measure)[0][-1])(par)
    log(f"  {time.time() - t0:.1f}s:  " + "  ".join(f"d/d{k} {float(v):+.3e}" for k, v in g.items()))

    write_csv([dict(day=10 * (k + 1), throat=float(thr[k]), inlet=float(inl[k]), lumen_area=float(area[k]),
                    wall_volume=float(vol[k])) for k in range(n_steps)], csv_path)
    t = np.arange(1, n_steps + 1) * 10.0
    plotting.panels([dict(t=t, series={"throat": np.asarray(thr), "inlet": np.asarray(inl)},
                          title="Lumen radius stretch", ylabel="r / r_ref (-)"),
                     dict(t=t, series={"lumen area": np.asarray(area), "wall volume": np.asarray(vol)},
                          title="Ratios to reference", ylabel="(-)")], fig)
    log.save(txt)


# ---------- mass redistribution: bent straight tube at homeostatic pressure ----------

def bending_redistribution(n_steps=100, theta=0.2, n_ramp=5, n=(3, 24, 60), L=60.0):
    """Straight half tube prestressed at p_hom (homeostasis everywhere), then both ends rotated by -+theta about y
    at unchanged pressure: collagen is produced on the stretched side (x < 0) and removed on the compressed side (x > 0);
    evaluated in the middle half, away from the clamped ends"""
    log = Log()
    txt, pvd, csv_path, fig = outputs("vessel/bending_redistribution", "bending.txt", "bending.pvd", "bending.csv", "bending.png")
    m = meshlib.half_tube(L=L, n=n)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"])
    xg = fem.gauss_points(m.X, m.conn)
    Q = meshlib.cylinder_basis(xg)
    Qj = jnp.asarray(Q)
    par = artery_par(p_gr=artery_par()["p_hom"])

    ends = np.concatenate([m.nodes["inlet"], m.nodes["outlet"]])
    sign = np.where(np.isin(ends, m.nodes["inlet"]), 1.0, -1.0)
    mid = np.flatnonzero(np.isclose(m.param[:, 2], 0.5))

    def bc(p, th=0.0):
        """u_y = 0 on y = 0; clamped end faces: u_x = 0, u_z = +-th x (end rotation about y). Distributed support:
        the lateral pressure resultant of the bent tube would otherwise load single nodes"""
        dofs = np.concatenate([3 * m.nodes["sym"] + 1, 3 * ends + 2, 3 * ends])
        vals = np.concatenate([np.zeros(len(m.nodes["sym"])), th * sign * m.X[ends, 0], np.zeros(len(ends))])
        i = np.argsort(dofs)
        return fem.BC(dofs[i], vals[i], p=p)

    bc_gr = [bc(par["p_gr"], theta * min(k / n_ramp, 1.0)) for k in range(1, n_steps + 1)]
    log(f"half tube r_i 5, t 1.3, L {L:.0f} mm: {m.n_elem} hex8, {sysm.n_dof} dofs, HCMM model E")
    log(f"prestress at {float(par['p_hom']) * 1e3:.0f} kPa, then G&R at the same pressure with both ends rotated "
        f"by {theta} rad (ramp over {n_ramp} steps): axial strain ~ +-{200 * theta * 6.3 / L:.1f} % at the outer wall")

    names = ("elastin", "circ", "axial", "helix+", "helix-")
    w = np.asarray(sysm.wdet)
    middle = (xg[..., 2] > L / 4) & (xg[..., 2] < 3 * L / 4)
    sides = {"stretched": (xg[..., 0] < 0) & middle, "compressed": (xg[..., 0] > 0) & middle}
    e_z = Qj[..., 2:3]
    M, S, rho0 = [], [], []
    series = meshlib.VTKSeries(pvd)

    def record(k, u, states):
        """mass after the commit of step k (day 10 k); stresses from the equilibrium state"""
        st = sysm.commit(u, states) if k else states
        rho = [np.asarray(c.rho) for c in st.constituents]
        if not k:
            rho0.extend(rho)
        M.append([[np.sum(w * r * mask) for r in rho] for mask in sides.values()])
        a = sysm.gauss_F(u) @ e_z
        a = a / jnp.linalg.norm(a, axis=-2, keepdims=True)
        s_zz = np.asarray(1e3 * (jnp.swapaxes(a, -1, -2) @ sysm.stress(u, states) @ a)[..., 0, 0])
        S.append([np.sum(w * s_zz * mask) / np.sum(w * mask) for mask in sides.values()])
        gauss = dict(rho_rel=st.rho_tot / st.rho_tot_0, J_g=det3(st.F_g), sigma_zz_kPa=s_zz,
                     **{f"rho_rel_{nm}": r / r0 for nm, r, r0 in zip(names[1:], rho[1:], rho0[1:])})
        series.write(10.0 * k, m.X, m.conn, *sysm.fields(u, states, gauss))

    t0 = time.time()
    (defl,), u, states = artery_simulate(m, sysm, Qj, par, n_steps=n_steps, bc=bc, bc_gr=bc_gr, log=log, final=True,
                                          measure=lambda u: (jnp.mean(jnp.asarray(u).reshape(-1, 3)[mid, 0]),),
                                          record=record)
    log(f"  done in {time.time() - t0:.1f}s, ParaView: {pvd.name} ({len(series.entries)} time steps)")

    M, S = np.array(M), np.array(S)
    col = M[:, :, 1:].sum(-1)
    col0 = col[0]
    d_side = 100 * (col / col0 - 1)
    d_tot = 100 * (col.sum(1) / col0.sum() - 1)
    moved = 100 * (col[:, 0] - col0[0]) / col0.sum()
    fam = 100 * (M[:, :, 1:] / M[0, :, 1:] - 1)
    log(f"\ncollagen mass change (%) in the middle half (z = {L / 4:.0f}-{3 * L / 4:.0f} mm), each side relative to its "
        "own start; 'gained' = stretched-side gain as % of the middle's collagen\n  (elastin has no turnover: constant)")
    for k in (0, 1, 5, 10, 25, 50, 100):
        if k <= n_steps:
            log(f"  day {10 * k:4d}  stretched {d_side[k, 0]:+7.2f}  compressed {d_side[k, 1]:+7.2f}  "
                f"total {d_tot[k]:+6.2f}  gained {moved[k]:+6.2f}   axial family {fam[k, 0, 1]:+7.2f} / "
                f"{fam[k, 1, 1]:+7.2f}   mean sigma_zz {S[k, 0]:6.1f} / {S[k, 1]:6.1f} kPa")
    log(f"  mid-span lateral deflection day {10 * n_ramp}: {float(defl[n_ramp - 1]):+.3f} mm, "
        f"day {10 * n_steps}: {float(defl[-1]):+.3f} mm")
    log("\nper family, day " + f"{10 * n_steps}: " + "   ".join(
        f"{nm} {fam[-1, 0, j]:+.2f} / {fam[-1, 1, j]:+.2f} %" for j, nm in enumerate(names[1:])))

    t = 10.0 * np.arange(n_steps + 1)
    write_csv([dict(day=t[k], collagen_stretched_pct=d_side[k, 0], collagen_compressed_pct=d_side[k, 1],
                    collagen_total_pct=d_tot[k], **{f"{nm}_{sd}_pct": fam[k, i, j] for j, nm in enumerate(names[1:])
                                                    for i, sd in enumerate(sides)},
                    sigma_zz_stretched_kPa=S[k, 0], sigma_zz_compressed_kPa=S[k, 1]) for k in range(n_steps + 1)],
              csv_path)
    plotting.panels([dict(t=t, series={"stretched half": d_side[:, 0], "compressed half": d_side[:, 1],
                                       "whole tube": d_tot}, title="Collagen mass change", ylabel="%"),
                     dict(t=t, series={f"{nm} {sd}": fam[:, i, j] for j, nm in enumerate(names[1:3])
                                       for i, sd in enumerate(sides)}, title="Per family", ylabel="%"),
                     dict(t=t, series={"stretched half": S[:, 0], "compressed half": S[:, 1]},
                          title="Mean axial stress", ylabel="sigma_zz (kPa)")], fig)
    log.save(txt)


def setpoint_patch(n_steps=200, delta=0.3, n=(3, 24, 40), L=40.0, w_z=4.0, w_theta=0.4):
    """Growth-only displacement: half tube prestressed to u = 0 at p_hom, loads unchanged afterwards; the collagen
    set point sigma_f_pre is lowered by delta in a patch (no elastic effect at the change), so all later u comes
    from growth and remodeling.  patch g = exp(-((z - L/2)/w_z)^2 - ((theta - pi/2)/w_theta)^2)"""
    log = Log()
    txt, pvd, csv_path, fig = outputs("vessel/setpoint_patch", *(f"setpoint_patch{s}" for s in (".txt", ".pvd", ".csv", ".png")))
    m = meshlib.half_tube(L=L, n=n)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"])
    xg = fem.gauss_points(m.X, m.conn)
    Qj = jnp.asarray(meshlib.cylinder_basis(xg))
    par = artery_par(p_gr=artery_par()["p_hom"])
    ends = np.concatenate([m.nodes["inlet"], m.nodes["outlet"]])

    def bc(p):
        """u_y = 0 on y = 0, clamped ends u_x = u_z = 0"""
        dofs = np.unique(np.concatenate([3 * m.nodes["sym"] + 1, 3 * ends, 3 * ends + 2]))
        return fem.BC(dofs, np.zeros(len(dofs)), p=p)

    theta = np.arctan2(xg[..., 1], xg[..., 0])
    g = np.exp(-((xg[..., 2] - L / 2) / w_z) ** 2 - ((theta - np.pi / 2) / w_theta) ** 2)
    insulted = []

    def insult(states):
        """lowered set point; the state before the insult is kept for the reference collagen mass"""
        insulted.append(states)
        cs = states.constituents
        return states.replace(constituents=[cs[0]] + [c.replace(sigma_f_pre=c.sigma_f_pre * (1 - delta * g))
                                                       for c in cs[1:]])

    def node(t, z, r):
        return int(np.flatnonzero(np.isclose(m.param[:, 1], t) & np.isclose(m.param[:, 2], z)
                                  & np.isclose(m.param[:, 0], r))[0])

    probes = {"patch": (node(0.5, 0.5, 0), node(0.5, 0.5, 1)), "far": (node(0.0, 0.5, 0), node(0.0, 0.5, 1))}
    w = np.asarray(sysm.wdet)
    in_patch = g > 0.5
    hist = []
    series = meshlib.VTKSeries(pvd)

    def record(k, u, states):
        """after the commit of step k (day 10 k)"""
        st = sysm.commit(u, states) if k else states
        x = m.X + np.asarray(u).reshape(-1, 3)
        row = dict(day=10.0 * k, max_u=float(np.linalg.norm(np.asarray(u).reshape(-1, 3), axis=1).max()))
        for name, (i, o) in probes.items():
            row[f"h_{name}"] = float(np.linalg.norm(x[o] - x[i]))
            row[f"r_in_{name}"] = float(np.linalg.norm(x[i, :2]))
        J_g = np.asarray(det3(st.F_g))
        rho = sum(np.asarray(c.rho) for c in st.constituents[1:])
        hist.append(row | dict(J_g_max=float(J_g.max()), collagen_patch=float(np.sum(w * rho * in_patch))))
        series.write(10.0 * k, m.X, m.conn, *sysm.fields(u, states, dict(
            J_g=J_g, rho_rel=st.rho_tot / st.rho_tot_0, setpoint=1 - delta * g,
            rho_rel_collagen=rho / sum(np.asarray(c.rho) for c in insulted[0].constituents[1:]))))

    log(f"half tube r_i 5, t 1.3, L {L:.0f} mm: {m.n_elem} hex8, {sysm.n_dof} dofs, HCMM model E, ends clamped")
    log(f"prestress at {float(par['p_hom']) * 1e3:.0f} kPa to u = 0; then loads unchanged, collagen set point sigma_f_pre "
        f"lowered by {100 * delta:.0f} % in a patch (w_z {w_z} mm, w_theta {w_theta} rad, top of the tube, mid-length)")
    t0 = time.time()
    artery_simulate(m, sysm, Qj, par, n_steps=n_steps, bc=bc, log=log, insult=insult, record=record,
                    measure=lambda u: (jnp.zeros(()),))
    log(f"  done in {time.time() - t0:.1f}s, ParaView: {pvd.name} ({len(series.entries)} time steps)")

    h0 = {k: hist[0][f"h_{k}"] for k in probes}
    r0 = {k: hist[0][f"r_in_{k}"] for k in probes}
    c0 = hist[0]["collagen_patch"]
    log("\n  day   max|u| [mm]   wall thickness patch / far   inner radius change patch / far [mm]   "
        "collagen in patch   J_g max")
    for k in (0, 1, 10, 25, 50, 100, 150, 200):
        if k <= n_steps:
            r = hist[k]
            log(f"  {r['day']:5.0f}   {r['max_u']:9.2e}   {r['h_patch'] / h0['patch']:10.4f} / {r['h_far'] / h0['far']:.4f}"
                f"          {r['r_in_patch'] - r0['patch']:+8.4f} / {r['r_in_far'] - r0['far']:+8.4f}"
                f"                {100 * (r['collagen_patch'] / c0 - 1):+6.2f} %   {r['J_g_max']:.4f}")

    write_csv(hist, csv_path)
    t = np.array([r["day"] for r in hist])
    plotting.panels([dict(t=t, series={"patch": np.array([r["h_patch"] for r in hist]) / h0["patch"],
                                       "far": np.array([r["h_far"] for r in hist]) / h0["far"]},
                          title="Wall thickness", ylabel="h / h(day 0) (-)"),
                     dict(t=t, series={"max |u|": np.array([r["max_u"] for r in hist]),
                                       "inner radius change, patch": np.array([r["r_in_patch"] - r0["patch"] for r in hist])},
                          title="Growth-induced displacement", ylabel="mm"),
                     dict(t=t, series={"collagen in patch": 100 * (np.array([r["collagen_patch"] for r in hist]) / c0 - 1)},
                          title="Collagen mass in the patch", ylabel="%")], fig)
    log.save(txt)
