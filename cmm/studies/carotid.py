"""Idealized carotid bifurcation (fem.mesh.carotid_bifurcation): homeostatic prestress of model E, tortuosity after
elastin loss with the Weiss 2020 wall."""

import hashlib
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import scipy.sparse as sp
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import fcmm
import fem
import hcmm
import materials
import plotting
from analysis import tortuosity as ti
from fem import mesh as meshlib
from fem.tensor3 import inv3
from rod import XG, Rod
from studies.common import RESULTS, Log, outputs, wall_stress, write_csv
from studies.model_e import artery_par, artery_simulate
from studies.weiss.model import KO, WALL_WEISS, weiss_par
from wall import assemble, principal_frame

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GEOMETRY = {  # CCA diameter D (mm); sinus (proximal ICA), distal ICA, ECA diameters / D; ICA, ECA angles from the CCA (deg)
    "bharadvaj": dict(D=6.3, sinus=1.11, ica=0.71, eca=0.57, angles=(25.0, 25.0)),
    "goubergrits": dict(D=6.45, sinus=1.09, ica=0.80, eca=0.72, angles=(33.8, 27.8)),
    "cad": dict(D=7.0, sinus=6 / 7, ica=6 / 7, eca=5 / 7, angles=(30.0, 20.0)),
}


def carotid_geometry(geometry, **override):
    """GEOMETRY[geometry] with the values of override that are not None; "bharadvaj": Bharadvaj et al. 1982 /
    Ku et al. 1985 (fem.mesh.carotid_bifurcation defaults), "goubergrits": means of 86 casts at 80 mmHg
    (Goubergrits et al. 2002, Table 3: D_c 6.45, D_i1 7.04, D_i2 5.17, D_e 4.63 mm, AI 33.8, AE 27.8 deg;
    SD 15-20 % of the diameters, 12 deg of the angles), "cad": the Creo model in .context/3dfiles (radii 3.5 / 3 /
    2.5 mm, ICA / ECA 30 / 20 deg, no sinus)"""
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


def junction_surface(geometry="goubergrits", D=None, angles=None, ica=None, eca=None, lengths=(20.0, 20.0, 20.0),
                     fillet=(1.6, 0.36), fillet_mm=None, voxel=None, smooth=5, name=None):
    """Bifurcation built as in CAD (Creo model .context/3dfiles): CCA (radius R, lengths[0] mm along -z) ending in a
    half-ball of radius R at the origin; ICA / ECA (radii ica R, eca R, lengths in mm) at angles from the CCA axis,
    placed so that their outer wall lines start on the CCA rim (+-R, 0, 0) at the same height (the half-ball equator);
    one body with rolling-ball rounds (fem.mesh.tube_junction): CCA / half-ball - branch rho = fillet[0] (R + r_b) / 2,
    ICA - ECA (flow divider) rho = fillet[1] (r_i + r_e) / 2, or fillet_mm (side, divider) in mm (CAD: 5, 1).
    Writes STL, VTU and the checks: watertight and manifold, far-field circles within 1%, max angle between
    adjacent face normals at the rounds (< ~30 deg), out-of-plane width (bulge), in-plane rounds against the circles
    tangent to both wall lines (and for "cad" against the round centres of the STEP), cross-sections, a view"""
    g = carotid_geometry(geometry, D=D, angles=angles, ica=ica, eca=eca)
    D, R = g["D"], 0.5 * g["D"]
    name = name or f"junction_{geometry}"
    e = axes(g["angles"])
    r = np.array([R, g["ica"] * R, g["eca"] * R])
    unit = lambda v: v / np.linalg.norm(v)
    rim = [None, np.array([R, 0.0, 0.0]), np.array([-R, 0.0, 0.0])]
    out = [None] + [unit(rim[j] - (rim[j] @ e[b]) * e[b]) for j, b in ((1, "ica"), (2, "eca"))]
    A = np.array([np.zeros(3)] + [rim[j] - r[j] * out[j] for j in (1, 2)])
    B = A + np.array(lengths)[:, None] * np.array([-e["inlet"], e["ica"], e["eca"]])
    side, divider = fillet_mm or (fillet[0] * (R + r[1:]) / 2, fillet[1] * r[1:].mean())
    rho = np.zeros((3, 3))
    rho[0, 1:] = rho[1:, 0] = side
    rho[1, 2] = rho[2, 1] = divider
    voxel = voxel or r.min() / 8
    k_max = rho.max()
    log = Log()
    stl, vtu, txt, sec, view = outputs("carotid/geometry", *(f"{name}{s}" for s in (
        ".stl", ".vtu", ".txt", "_sections.png", "_view.png")))
    t0 = time.time()
    V, T, F = meshlib.tube_junction(A, B, r, rho, voxel=voxel, smooth=smooth)
    log(f"tube junction ({geometry}: D {D} mm, ICA / ECA {g['angles'][0]:g} / {g['angles'][1]:g} deg, radii "
        f"{', '.join(f'{v:.3f}' for v in r)} mm, lengths {lengths} mm), branch outer walls from the CCA rim at the "
        f"half-ball equator; rounds CCA-ICA {rho[0, 1]:.2f}, CCA-ECA {rho[0, 2]:.2f}, ICA-ECA {rho[1, 2]:.2f} mm; "
        f"voxel {voxel:.3f} mm, {smooth} relaxation passes: {len(V)} vertices, {len(T)} triangles "
        f"({time.time() - t0:.0f}s)")

    def seg(x):
        AB = B - A
        t = np.clip(np.einsum("nij,ij->ni", x[:, None] - A, AB) / np.sum(AB * AB, 1), 0.0, 1.0)
        return np.linalg.norm(x[:, None] - (A + t[..., None] * AB), axis=2) - r, t

    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    key = np.sort(E, 1) @ np.array([len(V), 1])
    _, n_use = np.unique(key, return_counts=True)
    n_comp = connected_components(
        sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), (len(V), len(V))), directed=False)[0]
    vol = np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6
    log(f"  watertight / manifold: every edge in exactly 2 triangles {bool(np.all(n_use == 2))} (max {n_use.max()}), "
        f"consistently oriented {len(np.unique(E @ np.array([len(V), 1]))) == len(E)}, Euler characteristic "
        f"{len(V) - len(n_use) + len(T)} (2: closed, genus 0), components {n_comp}, enclosed volume {vol:.1f} mm^3")

    d, t = seg(V)
    for n_k in (1, 3):
        for i, tube in enumerate(("CCA", "ICA", "ECA")):
            L = np.linalg.norm(B[i] - A[i])
            far = (t[:, i] * L > voxel) & ((1 - t[:, i]) * L > voxel) & (np.delete(d, i, 1).min(1) > n_k * k_max)
            dist = np.linalg.norm(np.cross(V[far] - A[i], (B[i] - A[i]) / L), axis=1) / r[i] - 1
            log(f"  far field {tube} (other tubes > {n_k} max rho): {far.sum()} vertices, radius / r - 1 in "
                f"[{dist.min(initial=0):+.4f}, {dist.max(initial=0):+.4f}] (tolerance 1%)")

    log("  out-of-plane half-width max|y| / R over z / R (CCA R, no bulge: <= 1 up to the sinus-free branches): "
        + " ".join(f"{z:+.1f}: {np.abs(V[np.abs(V[:, 2] / R - z) < 0.05, 1]).max() / R:.3f}"
                   for z in (-2.0, -1.0, -0.5, 0.0, 0.3, 0.6, 1.0)))
    order = np.argsort(key, kind="stable").reshape(-1, 2) % len(T)
    N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip(np.sum(N[order[:, 0]] * N[order[:, 1]], 1), -1, 1)))
    dc, tc = seg(V[T].mean(1))
    ends = np.any(tc * np.linalg.norm(B - A, axis=1) > np.linalg.norm(B - A, axis=1) - 2 * voxel, 1)
    rnd = (np.sort(dc, 1)[:, 1] < k_max) & ~ends
    blend = rnd[order[:, 0]] | rnd[order[:, 1]]
    flat = ends[order[:, 0]] | ends[order[:, 1]]
    log(f"  adjacent face normals at the rounds (two tubes within {k_max:.2f} mm): max {angle[blend].max():.1f} deg, "
        f"99% {np.percentile(angle[blend], 99):.1f} deg (limit ~30 deg); tubes max {angle[~blend & ~flat].max():.1f} "
        f"deg; flat outlet edges {angle[flat].max():.1f} deg")

    rec =np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = N, V[T]
    with open(stl, "wb") as fh:
        fh.write(b"tube junction".ljust(80, b" ") + np.uint32(len(T)).tobytes() + rec.tobytes())

    planes = [(2, z * R) for z in (-1.0, 0.0, 0.6, 1.0, 1.3, 1.6, 2.0, 3.0)] + [(1, 0.0)]
    fig, axs = plt.subplots(3, 3, figsize=(13, 13))
    for (k, v), axp in zip(planes, axs.ravel()):
        u, w = [i for i in range(3) if i != k]
        sd = V[:, k] - v
        a_, b_ = T, np.roll(T, -1, 1)
        s0, s1 = sd[a_], sd[b_]
        cross = s0 * s1 < 0
        pts = V[a_] + (s0 / np.where(cross, s0 - s1, 1))[..., None] * (V[b_] - V[a_])
        two = cross.sum(1) == 2
        first = np.argsort(~cross[two], 1, kind="stable")[:, :2]
        segs = np.take_along_axis(pts[two], first[..., None], 1)[..., [u, w]]
        lim = np.array([V[:, [u, w]].min(0), V[:, [u, w]].max(0)])
        gu, gw = (np.linspace(lim[0, j], lim[1, j], 400) for j in range(2))
        GU, GW = np.meshgrid(gu, gw)
        Xp = np.zeros((GU.size, 3))
        Xp[:, u], Xp[:, w], Xp[:, k] = GU.ravel(), GW.ravel(), v
        axp.contour(GU, GW, F(Xp).reshape(GU.shape), [0.0], colors="k", linewidths=0.6)
        axp.add_collection(LineCollection(segs, colors="tab:red", linewidths=1.2))
        if k == 1:
            outline = np.c_[segs.reshape(-1, 2)[:, 0], np.zeros(2 * len(segs)), segs.reshape(-1, 2)[:, 1]]
        axp.set_aspect("equal")
        axp.set_title(f"{'xyz'[k]} = {v / R:+.1f} R" + (" (bifurcation plane)" if k == 1 else ""), fontsize=10)
        if k == 2:
            axp.set(xlim=(V[:, 0].min(), V[:, 0].max()), ylim=(V[:, 1].min(), V[:, 1].max()))
        else:
            axp.set(xlim=(-2.5 * D, 2.5 * D), ylim=(-1.5 * D, 3 * D))
    fig.suptitle(f"cross-sections: mesh (red), F = 0 (black); rounds {rho[0, 1]:.2f} / {rho[0, 2]:.2f} / "
                 f"{rho[1, 2]:.2f} mm, voxel {voxel:.3f} mm")
    fig.tight_layout()
    fig.savefig(sec, dpi=90)
    plt.close(fig)

    down = np.array([0.0, 0.0, -1.0])
    inner = [A[j] - r[j] * out[j] for j in (1, 2)]
    s_i = np.linalg.lstsq(np.c_[e["ica"], -e["eca"]], inner[1] - inner[0], rcond=None)[0][0]
    corners = [("CCA-ICA lateral", rim[1], down, e["ica"], rho[0, 1]),
               ("CCA-ECA lateral", rim[2], down, e["eca"], rho[0, 2]),
               ("ICA-ECA divider", inner[0] + s_i * e["ica"], e["ica"], e["eca"], rho[1, 2])]
    cad = {"CCA-ICA lateral": (8.5, 0.0, -1.3397), "CCA-ECA lateral": (-8.5, 0.0, -0.8816),
           "ICA-ECA divider": (-0.0025, 0.0, 7.9335)}
    for label, P, d1, d2, rc in corners:
        th = np.arccos(np.clip(d1 @ d2, -1, 1))
        ctr = P + rc / np.sin(th / 2) * unit(d1 + d2)
        q = outline - ctr
        arc = np.sum(q * unit(P - ctr), 1) / np.linalg.norm(q, axis=1) > np.cos((np.pi - th) / 2) + 1e-3
        arc &= np.linalg.norm(q, axis=1) < 2 * rc
        dev = np.linalg.norm(q[arc], axis=1) - rc
        log(f"  in-plane round {label}: circle radius {rc:.2f} mm tangent to both walls, centre "
            f"({ctr[0]:.3f}, {ctr[2]:.3f}) mm" + (f" (STEP round centre ({cad[label][0]:.3f}, {cad[label][2]:.3f}), "
                                                  f"{np.linalg.norm(ctr - cad[label]):.3f} mm apart)"
                                                  if geometry == "cad" else "")
            + f"; mesh outline on the arc: {arc.sum()} points, distance - radius in [{dev.min():+.3f}, "
              f"{dev.max():+.3f}] mm")

    fig = plt.figure(figsize=(14, 7))
    near = np.linalg.norm(V[T].mean(1), axis=1) < 2.5 * D
    light = np.array([0.4, -0.7, 0.6]) / np.linalg.norm([0.4, -0.7, 0.6])
    for j, (elev, azim) in enumerate(((10, -80), (35, -35))):
        ax3 = fig.add_subplot(1, 2, j + 1, projection="3d")
        shade = 0.2 + 0.8 * np.abs(N[near] @ light)
        ax3.add_collection3d(Poly3DCollection(V[T[near]], facecolors=np.c_[0.75 * shade, 0.45 * shade, 0.4 * shade]))
        ax3.set(xlim=(-2.5 * D, 2.5 * D), ylim=(-2.5 * D, 2.5 * D), zlim=(-2 * D, 3 * D))
        ax3.view_init(elev, azim)
        ax3.set_axis_off()
    fig.tight_layout()
    fig.savefig(view, dpi=100)
    plt.close(fig)
    meshlib.write_vtu(vtu, V, T, point_data={"fillet_mm": np.maximum(d.min(1), 0.0),
                                             "tube": np.argmin(np.abs(d), 1).astype(float)})
    log(f"  {stl.name}, {vtu.name} (fillet_mm: distance to the nearest tube, tube: 0 CCA, 1 ICA, 2 ECA), "
        f"{sec.name}, {view.name}")
    log.save(txt)


def bifurcation_tortuosity(geometry="goubergrits", D=None, angles=None, sinus=None, ica=None, eca=None,
                           lengths=(3.0, 8.0, 3.0), n_q=4, n_seam=5, n_r=2, noise=0.01,
                           ell=0.88, thickness="radius", kappa_junction=None, insult_day=10.0, n_hold=400, gamma=1.0,
                           ds=1.0, record_every=10, tol=1e-10, length=None, ends="isometric", half=False, model="hcmm",
                           fcmm_prestress=200, fillet=None, adapt=False, deposit="elastin", beta_junction=0.0, momentum=0.0,
                           pre_tol=1e-4, name=None,
                           folder="carotid/tortuosity", **kw):
    """Weiss 2020 wall (table S2) on the idealized bifurcation, wall thickness h / a of Weiss times the local radius
    (the mechanics are scale-free, so this is the mouse wall on a scaled carotid); thickness: carotid_bifurcation rule,
    "radius" (radius of the nearest tube, no out-of-plane thickening of the junction) or "curvature" (Laplace radius);
    kappa_junction: h / r on the junction patches instead (None: h / a everywhere; with the former C1 lumen the
    prestress stalled at h / a = 0.06 there; a thicker wall everywhere turns the response to the elastin loss into
    mass loss), prestressed at 120 mmHg (end_bc: isometric ends); ends "isometric" (kept) or "isotonic" (CCA
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
    geometry: carotid_geometry preset, D ... eca override it; fillet: CAD junction (carotid_bifurcation fillet, as
    bifurcation_rod); adapt: prestress with homeostatic fibre content (model_e.artery_simulate adapt: fibre masses,
    diagonal pair about the in-plane principal direction), and the emergent angle about that principal frame
    (wall.principal_frame: s_tt / s_zz -> sigma_c / sigma_a, diagonals at beta +- alpha_dep); deposit: artery_simulate
    ("elastin": beta from the elastin solve, "iterate": in the fixed point); beta_junction (deg, "iterate"): initial beta
    uniform in +-beta_junction on the junction elements (seed 1); momentum: prestress heavy-ball term; pre_tol: prestress
    tolerance on max|u| (mm); the converged prestress is cached under the mesh, parameter and prestress options
    (geometry/cache/prestress_<hash>.npz, not invalidated by code changes).
    Output per step: TI, deflection and axial force of the ICA tube (cuff to the distal end)"""
    g = carotid_geometry(geometry, D=D, angles=angles, sinus=sinus, ica=ica, eca=eca)
    D, angles = g["D"], g["angles"]
    log = Log()
    name = name or "bifurcation_tortuosity"
    txt, pvd, csv_path, fig, img, pvd_pre = outputs(folder, *(f"{name}{s}" for s in (
        ".txt", ".pvd", ".csv", ".png", "_end_Jg.png", "_prestress.pvd")))
    par = weiss_par(**kw)
    R, p0 = 0.5 * D, float(par["p_hom"])
    kappa = float(par["h"] / par["a"])
    t0 = time.time()
    margs = dict(D=D, angles=angles, sinus=g["sinus"], ica=g["ica"], eca=g["eca"], lengths=lengths, n_q=n_q,
                 n_seam=n_seam, n_r=n_r, kappa=kappa, kappa_junction=kappa_junction, thickness=thickness, half=half,
                 fillet=fillet)
    cache = RESULTS / "carotid" / "geometry" / "cache" / (
        f"mesh_{hashlib.sha1(repr(sorted(margs.items())).encode()).hexdigest()[:12]}.npz")
    sets = ("nodes", "faces", "elsets")
    if cache.exists():
        z = np.load(cache)
        m = meshlib.Mesh(z["X"], z["conn"], *({k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(s + ".")}
                                              for s in sets))
    else:
        m = meshlib.carotid_bifurcation(**margs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=m.X, conn=m.conn, **{f"{s}.{k}": v for s in sets for k, v in getattr(m, s).items()})
    sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"], element="hybrid")
    sysm.linear.reuse = False
    Q = jnp.asarray(fem.wall_basis(m.X, m.conn, m.nodes["inner"], m.nodes["outer"], [m.nodes["inlet"]],
                                   [m.nodes["ica"], m.nodes["eca"]]))
    length = length or max(np.sqrt(R * kappa * R), 1.2 * 0.5 * np.pi * R / n_q)
    bc = end_bc(m, angles)(p0)
    e_ica = axes(angles)["ica"]
    ica_nodes = np.intersect1d(np.unique(m.conn[m.elsets["ica"]]), m.nodes["inner"])
    s_ica, ring = np.unique(np.round(m.X[ica_nodes] @ e_ica, 6), return_inverse=True)
    log(f"idealized carotid bifurcation ({geometry}: D {D} mm, ICA / ECA {angles[0]:g} / {angles[1]:g} deg, diameters "
        f"sinus / ICA / ECA {g['sinus']} / {g['ica']} / {g['eca']} D, lengths {lengths} D), "
        f"h = {kappa:.3f} r (Weiss, {thickness}), {kappa_junction or kappa:.3f} r in the junction: {m.n_elem} hex8 (hybrid), {sysm.n_dof} dofs; Weiss table S2 wall, tr sigma "
        f"stimulus (nonlocal {length:.2f} mm), growth fixed, emergent collagen angle (gamma {gamma:g}); elastin c_e -> KO "
        f"on day {insult_day:g}, {n_hold} days held; elastin field {noise:g}; {ends} ends; ICA tube {len(s_ica)} rings"
        + (f"; CAD junction, rounds {fillet}" if fillet else "")
        + (f"; adapted prestress (fibre directions {deposit}"
           + (f", junction beta +-{beta_junction:g} deg" if deposit == "iterate" else "")
           + f", momentum {momentum:g})" if adapt else "")
        + f"; mesh {time.time() - t0:.0f}s")

    xg = np.asarray(fem.gauss_points(m.X, m.conn))
    rng = np.random.default_rng(0)
    w, ph = rng.standard_normal((256, 3)) / (ell * R), rng.uniform(0, 2 * np.pi, 256)
    xi = np.cos(xg @ w.T + ph).sum(-1)
    xi = (xi - xi.mean()) / np.abs(xi - xi.mean()).max()
    scale = jnp.asarray(1.0 + noise * xi)
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = Q @ jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])) @ jnp.swapaxes(Q, -1, -2)
    series_pre = meshlib.VTKSeries(pvd_pre, m)

    def pre_record(it, u, states, extra):
        F = np.asarray(sysm.gauss_F(u))
        fib = {}
        for j, q in enumerate(WALL_WEISS):
            if q.material == "fung":
                v = np.einsum("egij,egj->egi", F, np.asarray(states.constituents[j].material.M))
                fib[f"fiber_{q.name}"] = v / np.linalg.norm(v, axis=-1, keepdims=True)
        series_pre.write(it, m.X, m.conn, *sysm.fields(u, states, extra | fib))

    t0 = time.time()
    _, u, states = artery_simulate(m, sysm, Q, par, n_steps=0, bc=lambda p: bc, final=True, log=log, ds=ds,
                                   wall=WALL_WEISS, laws="wall", incompressible=True, growth="fixed", G_e=G_e,
                                   measure=lambda u: (jnp.zeros(()),),
                                   fields={"elastin.C10": par["elastin.C10"] * scale} if noise else None,
                                   anderson=2, pre_iter=200, pre_tol=pre_tol, pre_cache=cache.with_name(
                                       "prestress_" + hashlib.sha1(repr((cache.name, sorted(
                                           (k, np.asarray(v).tolist()) for k, v in par.items()), noise, ell, adapt,
                                           deposit, beta_junction, momentum, pre_tol)).encode()).hexdigest()[:12] + ".npz"),
                                   adapt=adapt, deposit=deposit, momentum=momentum,
                                   beta0=np.radians(beta_junction) * np.isin(np.arange(m.n_elem), m.elsets["junction"])[
                                       :, None] * np.random.default_rng(1).uniform(-1, 1, xg.shape[:2]),
                                   pre_record=pre_record)
    log(f"  prestress done in {time.time() - t0:.0f}s; ParaView: {pvd_pre.name} ({len(series_pre.entries)} iterations)")
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
        """(beta, s_c, s_a): principal frame of (s_tt - s_rr, s_zz - s_rr, s_tz) along the deformed wall directions,
        nonlocal mean (adapt; else 0, s_tt - s_rr, s_zz - s_rr)"""
        F, sig = sysm.gauss_F(u), sysm.stress(u, states)
        s = lambda a, b: jnp.einsum("...i,...ij,...j->...", a, sig, b) / jnp.sqrt(
            jnp.einsum("...i,...i->...", a, a) * jnp.einsum("...i,...i->...", b, b))
        rr = (lambda n: s(n, n))(jnp.linalg.solve(jnp.swapaxes(F, -1, -2), Q[..., :, 0][..., None])[..., 0])
        e = [jnp.einsum("...ij,...j->...i", F, Q[..., :, i]) for i in (1, 2)]
        tt, zz = (sysm.nonlocal_mean(s(v, v) - rr, length) for v in e)
        return principal_frame(tt, zz, sysm.nonlocal_mean(s(*e), length)) if adapt else (0.0, tt, zz)

    _, s_c, s_a = in_plane(u, states)
    ratio_h = s_c / s_a
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
        beta, s_c, s_a = in_plane(u, states)
        al = jnp.asarray(np.arctan(np.tan(alpha) * (np.maximum(s_c, 0.0) / np.maximum(s_a, 1e-12) / ratio_h)
                                   ** gamma))[..., None]
        beta = jnp.asarray(beta)[..., None] if adapt else 0.0
        for d, sign in (("diag+", 1.0), ("diag-", -1.0)):
            c = cs[names.index(d)]
            cs[names.index(d)] = set_part(c, production=part(c).production.replace(
                M_dep=jnp.sin(al + sign * beta) * e_t + sign * jnp.cos(al + sign * beta) * e_z))
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
                    s_sinus=(0.3, 0.8, 1.8), junction="wall", fillet=(1.6, 0.36), junction_voxel=None,
                    cuffs=(0.6, 1.5, 1.4), kappa_junction=0.10, tie_EA=2.0, el=0.5, n_ring=16, noise=0.01,
                    ell=0.88, insult_day=10.0, n_hold=400, gamma=1.0, ds=1.0, length=None, record_every=10, name=None,
                    folder="carotid/rod", **kw):
    """bifurcation_tortuosity on the reduced model (rod.Rod): CCA, ICA, ECA centerlines as 3D rods of elements of
    about el mm, joined rigidly at the origin, ends clamped (isometric), wall thickness h / a of Weiss.
    junction "wall": the bifurcation as in CAD (junction_surface): CCA ending in a half-ball of radius R at the origin,
    ICA (radius sinus R until s2, -> ica R at s3) and ECA (eca R) with their outer wall lines from the CCA rim
    (branch rods start beside the junction node, rod.Rod offsets), rounds rho = fillet[0] (R + r_b) / 2 on CCA /
    half-ball - branch, fillet[1] (r_i + r_e) / 2 on the flow divider; the junction zone (each tube up to 0.5 R beyond
    its rounds) is a junction wall (rod.Rod.attach_junction): membrane material points on that surface (vertices of
    fem.mesh.tube_junction, voxel junction_voxel, default min r / 5; of the y >= 0 half, mirrored) with the Weiss
    wall and G&R, moved by the rod
    sections (weights exp(-|d_i| / 0.1 R), d_i distance to tube i), pressure on the junction lumen; one rod element per
    tube spans the zone and keeps only its kinematics.
    junction "ties": ICA 0.9 R -> sinus R on [0, s1], branches from the origin, kappa_junction r from the junction to
    the cuffs (cuffs D; blended over 0.2 D), a triangle of elastic bars (tie_EA in N, 0: none) between the branch
    centres at the cuffs (rod.Rod.tie) (sections at the junction bend and grow without bound);
    elastin c_e -> KO on insult_day, n_hold steps of ds days; emergent collagen angle (gamma); tr sigma stimulus,
    nonlocal over length (default max(sqrt(R h), 1.2 x ring point spacing)); the same 3D elastin field as the FE
    (seed 0, 256 Fourier modes, correlation length ell R). Output per step: TI, deflection and axial force of the
    ICA beyond its cuff, J_g, alpha; ParaView: deformed walls of the three branches (and {name}_junction.pvd: the
    junction wall); geometry: carotid_geometry preset, D ... eca override it."""
    g = carotid_geometry(geometry, D=D, angles=angles, sinus=sinus, ica=ica, eca=eca)
    D, angles, sinus, ica, eca = g["D"], g["angles"], g["sinus"], g["ica"], g["eca"]
    log = Log()
    name = name or "bifurcation_rod"
    txt, pvd, pvd_j, csv_path, fig = outputs(folder, *(f"{name}{s}" for s in (
        ".txt", ".pvd", "_junction.pvd", ".csv", ".png")))
    par = weiss_par(**kw)
    R, p0 = 0.5 * D, float(par["p_hom"])
    kappa = float(par["h"] / par["a"])
    length = length or max(np.sqrt(R * kappa * R), 1.2 * 2 * np.pi * R / n_ring)
    e = axes(angles)
    keys = ("inlet", "ica", "eca")
    unit = lambda v: v / np.linalg.norm(v)
    s1, s2, s3 = (v * D for v in s_sinus)
    sm = lambda s, lo, hi: (lambda t: t * t * (3 - 2 * t))(np.clip((s - lo) / (hi - lo), 0.0, 1.0))
    wall = junction == "wall"
    dirs = np.array([-e["inlet"], e["ica"], e["eca"]])
    start, zone = np.zeros((3, 3)), np.zeros(3)
    if wall:
        r_j = np.array([R, sinus * R, eca * R])
        for i in (1, 2):
            rim = np.array([R if i == 1 else -R, 0.0, 0.0])
            start[i] = rim - r_j[i] * unit(rim - (rim @ dirs[i]) * dirs[i])
        rho = np.zeros((3, 3))
        rho[0, 1:] = rho[1:, 0] = fillet[0] * (R + r_j[1:]) / 2
        rho[1, 2] = rho[2, 1] = fillet[1] * r_j[1:].mean()
        L_t = 3 * R + 2 * rho.max()
        t0 = time.time()
        Vj, Tj, _ = meshlib.tube_junction(start, start + L_t * dirs, r_j, rho, voxel=junction_voxel or r_j.min() / 5)
        s_t = np.einsum("nij,ij->ni", Vj[:, None] - start, dirs)
        d_t = np.linalg.norm(Vj[:, None] - start - np.clip(s_t, 0, L_t)[..., None] * dirs, axis=2) - r_j
        rnd = d_t.min(1) > 1e-3 * r_j.min()
        zone = np.array([s_t[rnd & (np.abs(d_t[:, i]) < rho.max()), i].max() + 0.5 * R for i in range(3)])
        if zone[1] > s2:
            s2, s3 = zone[1], s3 + zone[1] - s2
        log(f"junction surface ({len(Vj)} vertices, {time.time() - t0:.0f}s): rounds CCA-ICA {rho[0, 1]:.2f}, CCA-ECA "
            f"{rho[0, 2]:.2f}, ICA-ECA {rho[1, 2]:.2f} mm; junction zone CCA / ICA / ECA {zone[0]:.2f} / {zone[1]:.2f} / "
            f"{zone[2]:.2f} mm along the tubes")
    X, conn, branch, s_node, off = [np.zeros(3)], [], [], [0.0], []
    for b, L_b in enumerate(lengths):
        L = L_b * D
        if zone[b]:
            s_b = np.r_[zone[b], np.linspace(zone[b], L, max(2, int(np.ceil((L - zone[b]) / el))) + 1)[1:]]
        else:
            s_b = L * np.arange(1, max(2, round(L / el)) + 1) / max(2, round(L / el))
        i0, n = len(X), len(s_b)
        if b == 0:
            X += [s * dirs[0] for s in s_b[::-1]]
            s_node += list(s_b[::-1])
            conn += [(i0 + i, i0 + i + 1) for i in range(n - 1)] + [(i0 + n - 1, 0)]
            off += [np.zeros((2, 3))] * n
        else:
            X += [start[b] + s * dirs[b] for s in s_b]
            s_node += list(s_b)
            conn += [(0, i0)] + [(i0 + i, i0 + i + 1) for i in range(n - 1)]
            off += [np.array([start[b], np.zeros(3)])] + [np.zeros((2, 3))] * (n - 1)
        branch += [b] * n
    X, conn, branch, s_node, off = np.array(X), np.array(conn), np.array(branch), np.array(s_node), np.array(off)
    s_end = np.where(conn == 0, 0.0, s_node[conn])
    active = s_end.mean(1) > zone[branch]
    ends = [1, conn[branch == 1][-1, 1], conn[branch == 2][-1, 1]]
    s_g = s_end[:, :1] * (1 - XG) + s_end[:, 1:] * XG
    ica_r = (lambda s: sinus * R + (ica - sinus) * R * sm(s, s2, s3)) if wall else \
        (lambda s: 0.9 * R + (sinus - 0.9) * R * sm(s, 0, s1) + (ica - sinus) * R * sm(s, s2, s3))
    radius = lambda s: np.select([branch[:, None] == 0, branch[:, None] == 1], [np.full(s.shape, R), ica_r(s)], eca * R)
    A_end, A = radius(s_end), radius(s_g)
    s_cuff = D * np.array(cuffs)[branch][:, None]
    kap = kappa if wall else kappa_junction + (kappa - kappa_junction) * sm(s_g, s_cuff - 0.2 * D, s_cuff)
    rod = Rod(X, conn, branch, A_end, kap * A, hcmm, rigid=(0,), clamped=ends, n_ring=n_ring, offsets=off,
              active=active)
    if wall:
        own = np.argmin(np.abs(d_t), 1)
        sel = s_t[np.arange(len(Vj)), own] <= zone[own]
        ids = np.flatnonzero(sel & (Vj[:, 1] >= 0))
        nrm = np.cross(Vj[Tj[:, 1]] - Vj[Tj[:, 0]], Vj[Tj[:, 2]] - Vj[Tj[:, 0]])
        dA = np.bincount(Tj.ravel(), np.repeat(0.5 * np.linalg.norm(nrm, axis=1) / 3, 3), len(Vj))
        dA *= np.where(np.abs(Vj[:, 1]) < 1e-9 * R, 0.5, 1.0)
        N = np.stack([np.bincount(Tj.ravel(), np.repeat(nrm[:, c], 3), len(Vj)) for c in range(3)], -1)
        w_j = np.exp(-(np.abs(d_t) - np.abs(d_t).min(1, keepdims=True)) / (0.1 * R))
        w_j /= w_j.sum(1, keepdims=True)
        L0, T0 = (np.asarray(v) for v in rod.geo[:2])
        elems = np.zeros((len(Vj), 3), int)
        for b in range(3):
            eb = np.flatnonzero(branch == b)
            s_e = np.einsum("nei,ei->ne", Vj[:, None] - rod.Xe[eb, 0], T0[eb])
            elems[:, b] = eb[np.argmin(np.maximum(np.maximum(-s_e, s_e - L0[eb]), 0.0), 1)]
        cid = np.unique(np.floor(Vj[ids] / (0.25 * R)).astype(int), axis=0, return_inverse=True)[1].ravel()
        area = np.bincount(cid, dA[ids])
        cent = np.stack([np.bincount(cid, dA[ids] * Vj[ids, c]) for c in range(3)], -1) / area[:, None]
        order = np.lexsort((np.linalg.norm(Vj[ids] - cent[cid], axis=1), cid))
        rep = ids[order[np.r_[True, cid[order][1:] != cid[order][:-1]]]]
        Nc = np.stack([np.bincount(cid, N[ids, c]) for c in range(3)], -1)
        Nc /= np.linalg.norm(Nc, axis=1, keepdims=True)
        Ez = w_j[rep] @ np.array([e["inlet"], e["ica"], e["eca"]])
        Ez -= np.sum(Ez * Nc, 1, keepdims=True) * Nc
        Ez += 1e-3 * np.cross([0.0, 1.0, 0.0], Nc)
        Ez /= np.linalg.norm(Ez, axis=1, keepdims=True)
        bnd = [np.flatnonzero((branch == b) & np.isclose(s_end[:, 0], zone[b]))[0] for b in range(3)]
        M = np.array([1.0, -1.0, 1.0])
        rod.attach_junction(np.r_[Vj[rep], M * Vj[rep]], np.r_[Nc, M * Nc], np.r_[Ez, M * Ez], np.r_[area, area],
                            np.tile(kappa * r_j[own[rep]], 2), np.r_[elems[rep], elems[rep]], np.r_[w_j[rep], w_j[rep]],
                            [(bnd[0], 0, -1.0), (bnd[1], 0, 1.0), (bnd[2], 0, 1.0)])
        ids = np.flatnonzero(sel)
        idx = np.full(len(Vj), -1)
        idx[ids] = np.arange(len(ids))
        tri_j = idx[Tj[np.all(sel[Tj], 1)]]
        out_j = (Vj[ids], elems[ids], w_j[ids])
        near = cKDTree(rod.X_points[rod.n_rp:]).query(Vj[ids])[1]
    elif tie_EA:
        nodes_b = [np.setdiff1d(conn[branch == b].ravel(), [0]) for b in range(3)]
        cuff = [n[np.argmin(np.abs(np.abs(X[n] @ e[k]) - c * D))] for n, k, c in zip(nodes_b, keys, cuffs)]
        rod.tie([(cuff[1], cuff[2]), (cuff[0], cuff[1]), (cuff[0], cuff[2])], tie_EA)
    shp = (len(rod.X_points), 1)
    rng = np.random.default_rng(0)
    w, ph = rng.standard_normal((256, 3)) / (ell * R), rng.uniform(0, 2 * np.pi, 256)
    xi = np.cos(rod.X_points @ w.T + ph).sum(-1)
    xi = ((xi - xi.mean()) / np.abs(xi - xi.mean()).max()).reshape(shp)
    scale = jnp.asarray(1.0 + noise * xi)
    ica_nodes = np.concatenate([[0], conn[branch == 1][:, 1]])
    ica_nodes = ica_nodes[X[ica_nodes] @ e["ica"] >= cuffs[1] * D]
    log(f"idealized carotid bifurcation as 3D rods ({geometry}: D {D} mm, ICA / ECA {angles[0]:g} / {angles[1]:g} deg, lengths "
        f"{lengths} D, radii ICA {'' if wall else '0.9-'}{sinus}-{ica} R, ECA {eca} R), h = {kappa:.3f} r (Weiss)"
        + (f", junction wall {rod.junction['n']} points" if wall else f", {kappa_junction} r up to the cuffs {cuffs} D")
        + f": {', '.join(f'{k} {np.sum(branch == b)}' for b, k in enumerate(('CCA', 'ICA', 'ECA')))} elements x 3 "
        f"sections x {n_ring} ring points ({np.sum(~active)} in the junction zone), {len(rod.free)} DOFs; rigid junction "
        f"node, clamped isometric ends; tr sigma stimulus (nonlocal {length:.2f} mm), emergent collagen angle (gamma "
        f"{gamma:g}); c_e -> KO on day {insult_day:g}, {n_hold} days held; elastin field {noise:g}")

    t0 = time.time()
    G_t, G_z = float(par["elastin.G_theta"]), float(par["elastin.G_z"])
    G_e = jnp.broadcast_to(jnp.diag(jnp.array([1 / (G_t * G_z), G_t, G_z])), shp + (3, 3))
    Q = jnp.broadcast_to(jnp.eye(3), G_e.shape)
    states, it, du = rod.prestress(lambda G: assemble(G, Q, par, WALL_WEISS, ds=ds, laws="wall", incompressible=True,
                                                      growth="fixed", fields={"elastin.C10": par["elastin.C10"] * scale}),
                                   G_e, p0, anderson=5, tol=1e-6)
    ties = bool(tie_EA) and not wall
    log(f"  prestress: {it} iterations, max|q - q_0| {du:.1e}  ({time.time() - t0:.0f}s)"
        + (f"; junction ties ICA-ECA / CCA-ICA / CCA-ECA {', '.join(f'{1e3 * v:.1f}' for v in rod.tie_forces(rod.q0))} mN"
           if ties else ""))
    q = rod.q0
    s_tt, s_zz = rod.stresses(q, states)
    sw = rod.nonlocal_mean(s_tt + s_zz, length)
    states = states.replace(sigma_w=sw, sigma_w_h=sw)
    ratio_h = rod.nonlocal_mean(s_tt, length) / rod.nonlocal_mean(s_zz, length)
    names = [c.name for c in WALL_WEISS]
    alpha = par["diag+.alpha"]
    e_t, e_z = jnp.array([0.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    angle = lambda M: np.degrees(np.arccos(np.abs(np.asarray(M @ e_z))))
    J_g = lambda st: np.asarray(jax.vmap(jax.vmap(hcmm.J_target))(st)).ravel()
    series, series_j = meshlib.VTKSeries(pvd), meshlib.VTKSeries(pvd_j)
    X_ref = rod.wall(q, states)
    Xj_ref = out_j[0] if wall else None
    n_rp = rod.n_rp
    last_ica = (3 * int(np.flatnonzero(branch == 1)[-1]) + 2) * n_ring + np.arange(n_ring)
    real = np.r_[np.repeat(active, 3 * n_ring), np.ones(len(rod.X_points) - n_rp, bool)]
    junc = np.r_[np.zeros(n_rp, bool), np.ones(len(rod.X_points) - n_rp, bool)]

    def record(day, q, st):
        Xw, cw = rod.wall(q, st)
        s_tt, s_zz = (np.asarray(v).ravel() for v in rod.stresses(q, st))
        diag = st.constituents[names.index("diag+")]
        fields = dict(J_g=J_g(st), sigma_tt_kPa=1e3 * s_tt, sigma_zz_kPa=1e3 * s_zz,
                      c_e_rel=np.asarray(st.constituents[0].material.C10 / par["elastin.C10"]).ravel(),
                      elastin_field=xi.ravel(), alpha_diag_deg=angle(diag.material.M).ravel(),
                      alpha_dep_deg=angle(diag.production.M_dep).ravel(),
                      wall_stress_rel=np.asarray(st.sigma_w / st.sigma_w_h).ravel())
        fr = {k: np.repeat(v[:n_rp], 2) for k, v in fields.items()} | {"branch": np.repeat(branch, 6 * n_ring)}
        series.write(day, X_ref[0], cw, fr | {"u": Xw - X_ref[0]}, {k: v[cw].mean(1) for k, v in fr.items()})
        if wall:
            fj = {k: v[n_rp:][near] for k, v in fields.items()}
            series_j.write(day, Xj_ref, tri_j, fj | {"u": rod.map_points(q, *out_j) - Xj_ref},
                           {k: v[tri_j].mean(1) for k, v in fj.items()})

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
        F = np.asarray(rod.F(q, states)).reshape(-1, 3, 3)
        a_end = float(np.asarray(rod.expand(q))[last_ica[0] // (3 * n_ring), 12:14] @ np.array([1 - XG[2], XG[2]]))
        s_zz = np.asarray(rod.stresses(q, states)[1]).ravel()
        N = float(np.sum(s_zz[last_ica] * kappa * A.ravel()[last_ica[0] // n_ring] * F[last_ica, 0, 0] * a_end * 2
                         * np.pi / n_ring))
        jg = J_g(states)
        al_f = angle(states.constituents[names.index("diag+")].material.M).ravel()[real]
        rows.append(dict(day=ds * k, TI_ica=ti(c), deflection_ica=np.linalg.norm(off, axis=1).max() / R,
                         out_of_plane_ica=np.abs(off[:, 1]).max() / R, J_g_mean=jg[real].mean(),
                         J_g_max=jg[real & ~junc].max(), J_g_max_junction=jg[junc].max() if wall else 0.0,
                         alpha_mean=al_f.mean(), alpha_min=al_f.min(), alpha_max=al_f.max(), axial_force_ica_mN=1e3 * N,
                         **({f"tie_{t}_mN": 1e3 * v for t, v in zip(("divider", "cca_ica", "cca_eca"),
                                                                    rod.tie_forces(q))} if ties else {})))
        if k % record_every == 0 or k == n_steps:
            record(ds * k, q, states)
        if k % max(n_steps // 20, 1) == 0:
            r = rows[-1]
            log(f"  day {r['day']:5.0f}  ICA: TI {r['TI_ica']:.2e}  deflection/R {r['deflection_ica']:.3f} (out of plane "
                f"{r['out_of_plane_ica']:.3f})  axial force {r['axial_force_ica_mN']:.1f} mN   J_g mean "
                f"{r['J_g_mean']:.4f} max rods {r['J_g_max']:.4f}"
                + (f" junction {r['J_g_max_junction']:.4f}" if wall else "")
                + f"   alpha mean {r['alpha_mean']:.2f} ({r['alpha_min']:.2f}-{r['alpha_max']:.2f}) deg"
                + (f"  ties {r['tie_divider_mN']:.0f} / {r['tie_cca_ica_mN']:.0f} / {r['tie_cca_eca_mN']:.0f} mN"
                   if ties else "")
                + f"  (Newton {it}, {time.time() - t0:.0f}s)")
    log(f"  ParaView: {pvd.name}{f' + {pvd_j.name}' if wall else ''} ({len(series.entries)} time steps)")
    write_csv(rows, csv_path)
    col = lambda key: np.array([r[key] for r in rows])
    t = col("day")
    plotting.panels([dict(t=t, series={"ICA": col("TI_ica")}, title="Tortuosity index", ylabel="(-)"),
                     dict(t=t, series={"ICA": col("deflection_ica")}, title="Lateral deflection / R", ylabel="(-)"),
                     dict(t=t, series={"ICA": col("axial_force_ica_mN")}, title="Axial force", ylabel="mN"),
                     dict(t=t, series={"mean": col("J_g_mean"), "max rods": col("J_g_max")}
                          | ({"max junction": col("J_g_max_junction")} if wall else {}), title="Volume growth J_g",
                          ylabel="(-)"),
                     dict(t=t, series={"mean": col("alpha_mean"), "min": col("alpha_min"), "max": col("alpha_max")},
                          title="Diagonal collagen angle", ylabel="deg")], fig)
    log.save(txt)
