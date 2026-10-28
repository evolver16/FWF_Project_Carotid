"""Video of a tube G&R series (.pvd): deformed tube coloured by a field and the diagonal fiber angle on the convex and
concave side of the bend; PNG frames for LaTeX (animate package), a .tex snippet and a GIF preview.

python -m analysis.video <series.pvd> [--field J_g] [--fps 8] [--hold 3]"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from fem.mesh import HEX_FACES, read_pvd, read_vtu


def artery_video(pvd, out=None, field="J_g", angle="alpha_diag_deg", alpha_h=38.1, fps=8, hold=3.0, dpi=150):
    """<out>/frame_<k>.png, <out>/<name>.tex (animateinline, first and last frame held for hold seconds) and
    <out>/<name>.gif, out default videos/<pvd stem> next to the .pvd, name = out folder name; LaTeX: copy the folder
    next to the document and \\input{<name>/<name>.tex};
    left: outer surface of the tube (a half tube mirrored at y = 0) coloured by field, fixed range over all frames;
    right: the two diagonal families at +-angle from the axial direction (axial horizontal, circumferential vertical)
    on the convex and concave side (along the bending direction) of the cross-section with the largest final deflection; dashed: alpha_h"""
    pvd = pathlib.Path(pvd)
    out = pathlib.Path(out) if out else pvd.parent / "videos" / pvd.stem
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("frame_*.png"):
        f.unlink()
    series = read_pvd(pvd)
    frames = [read_vtu(f) for _, f in series]
    conn = frames[0][1]
    X = frames[0][2].get("X_ref", frames[0][0])
    shape = lambda fr: fr[0] if "X_ref" in fr[2] else X + fr[2]["u"]

    faces = np.concatenate([conn[:, f] for f in HEX_FACES.values()])
    _, idx, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    faces = faces[idx[cnt == 1]]
    faces = faces[np.hypot(*X[faces][..., :2].mean(1).T) > np.median(np.hypot(*X[faces][..., :2].mean(1).T))]

    half = X[:, 1].min() > -1e-9 * np.abs(X).max()
    x_end = shape(frames[-1])
    _, ring = np.unique(np.round(X[:, 2], 9), return_inverse=True)
    c = np.stack([np.bincount(ring, x_end[:, i]) / np.bincount(ring) for i in (0, 1)], -1)
    if half:
        c[:, 1] = 0.0
    k = np.argmax(np.linalg.norm(c, axis=1))
    station = ring == k
    bow = c[k] / (np.linalg.norm(c[k]) or 1.0)
    r = np.hypot(X[:, 0], X[:, 1])
    towards = X[:, :2] @ bow / r
    sides = {"convex": station & (towards > 0.9), "concave": station & (towards < -0.9)}
    ring_nodes = np.flatnonzero(station & (r > np.median(r[station])))
    ring_nodes = ring_nodes[np.argsort(np.arctan2(X[ring_nodes, 1], X[ring_nodes, 0]))]

    mirror = np.array([1.0, -1.0, 1.0])
    polys = [(np.concatenate([x[faces], x[faces] * mirror]) if half else x[faces])[..., [2, 1, 0]]
             for x in map(shape, frames)]
    vals = [np.tile(fr[2][field][faces].mean(1), 2 if half else 1) for fr in frames]
    lo = np.min([p.reshape(-1, 3).min(0) for p in polys], axis=0)
    hi = np.max([p.reshape(-1, 3).max(0) for p in polys], axis=0)
    norm = plt.Normalize(min(v.min() for v in vals), max(v.max() for v in vals))
    cmap = plt.get_cmap("viridis")

    def fibers(ax, a, title):
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
        for tip, start, label, pos, ha in (((1.25, 0.5), (1.02, 0.5), "axial", (1.13, 0.56), "center"),
                                           ((0.5, 1.25), (0.5, 1.02), "circumferential", (0.56, 1.13), "left")):
            ax.annotate("", xy=tip, xytext=start, xycoords="axes fraction", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k"))
            ax.text(*pos, label, transform=ax.transAxes, ha=ha, va="bottom" if ha == "center" else "center", fontsize=8)
        for ang, style in ((alpha_h, dict(color="0.7", lw=0.8, ls="--")), (a, dict(color="C3", lw=1.4))):
            for sign in (1.0, -1.0):
                d = np.array([np.cos(np.radians(sign * ang)), np.sin(np.radians(sign * ang))])
                n = np.array([-d[1], d[0]])
                for c in np.linspace(-1.2, 1.2, 11):
                    p = 0.5 + c * n
                    ax.plot([p[0] - 2 * d[0], p[0] + 2 * d[0]], [p[1] - 2 * d[1], p[1] + 2 * d[1]], **style)
        ax.text(0.5, -0.06, f"{title}: α = {a:.1f}°\n({'more axial' if a < alpha_h else 'more circumferential'})",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)

    for k, ((t, _), fr, poly, val) in enumerate(zip(series, frames, polys, vals)):
        fig = plt.figure(figsize=(12, 4.2))
        ax = fig.add_axes([-0.06, -0.05, 0.74, 1.1], projection="3d", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(poly, facecolors=cmap(norm(val)), edgecolors="none", zorder=1))
        x = shape(fr)
        ring = np.concatenate([x[ring_nodes], (x[ring_nodes] * mirror)[::-1] if half else x[ring_nodes[:1]]])[:, [2, 1, 0]]
        ax.plot(*ring.T, color="k", lw=1.5, zorder=2)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=25, azim=-70)
        ax.set_xlabel("z (mm)"); ax.set_zlabel("x (mm)"); ax.set_yticks([])
        fig.text(0.02, 0.92, f"day {t:.0f}", fontsize=12)
        fig.text(0.02, 0.05, "black ring: section of the fiber panels; α from the tube axis (0° axial, 90° circumferential), "
                 f"red: current, dashed: healthy {alpha_h:g}°", fontsize=8)
        cax = fig.add_axes([0.66, 0.2, 0.012, 0.6])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=field)
        for (name, sel), rect in zip(sides.items(), ([0.74, 0.6, 0.18, 0.28], [0.74, 0.14, 0.18, 0.28])):
            fibers(fig.add_axes(rect), float(fr[2][angle][sel].mean()), name)
        fig.savefig(out / f"frame_{k:03d}.png", dpi=dpi)
        plt.close(fig)

    n = len(frames)
    rates = [1 / hold] + [fps] * (n - 2) + [1 / hold]
    lines = [f"\\includegraphics[width=\\linewidth]{{{out.name}/frame_{k:03d}.png}}" for k in range(n)]
    body = [lines[0]] + [f"\\newframe[{rates[k]:.4g}]\n{lines[k]}" if k in (1, n - 1) else f"\\newframe\n{lines[k]}"
                         for k in range(1, n)]
    (out / f"{out.name}.tex").write_text(
        "% \\usepackage{animate}, \\usepackage{graphicx}; folder " + out.name + "/ next to the document, "
        "\\input{" + out.name + "/" + out.name + ".tex}\n"
        f"\\begin{{animateinline}}[controls,loop,autoplay]{{{rates[0]:.4g}}}\n" + "\n".join(body) + "\n\\end{animateinline}\n")
    images = [Image.open(out / f"frame_{k:03d}.png").convert("P", palette=Image.ADAPTIVE) for k in range(n)]
    images[0].save(out / f"{out.name}.gif", save_all=True, append_images=images[1:],
                   duration=[1000 / r for r in rates], loop=0)
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("pvd")
    a.add_argument("--field", default="J_g")
    a.add_argument("--fps", type=float, default=8)
    a.add_argument("--hold", type=float, default=3.0)
    args = a.parse_args()
    print(artery_video(args.pvd, field=args.field, fps=args.fps, hold=args.hold))
