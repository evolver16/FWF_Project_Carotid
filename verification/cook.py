"""Cook's membrane (plane strain, nearly / fully incompressible neo-Hookean): tapered panel (0,0)-(48,44)-(48,60)-(0,44)
clamped at x = 0, uniform dead shear traction tau = F / 16 on x = 48; vertical displacement of the tip (48, 60).
Bending with skewed elements: the standard locking benchmark. Reference: Richardson extrapolation of hex8 hybrid,

    v_inf = v_h + (v_h - v_2h) / (2^p - 1),   p = log2((v_4h - v_2h) / (v_2h - v_h))

fbar and hybrid (hex8, tet10) must approach it, standard hex8 locks. mu = 2 C10 = 80.194, K / mu = 5000 (fbar, standard)
or incompressible (hybrid), F = 32 per unit thickness, plane strain via u_z = 0.
Run: python -m verification.cook
"""

import pathlib
import time

import numpy as np

import fem
import materials
from fem import mesh as meshlib

MU, RATIO, FORCE, T = 80.194, 5000.0, 32.0, 1.0
OUT = pathlib.Path(__file__).parent / "results"


def tip_displacement(n, etype, element, steps=4):
    def mapping(s):
        x = 48.0 * s[:, 0]
        y = (1 - s[:, 1]) * 44.0 * s[:, 0] + s[:, 1] * (44.0 + 16.0 * s[:, 0])
        return np.stack([x, y, T * s[:, 2]], axis=1)

    m = meshlib.structured((n, n, 1), mapping, etype)
    sysm = fem.System(m, element=element)
    material = materials.NeoHookeanInc(MU / 2) if element == "hybrid" else materials.NeoHookean(MU / 2, RATIO * MU)
    states = fem.broadcast_state(material, m.n_elem, sysm.n_gp)

    face = m.element.face
    faces = m.faces[(0, 1)]
    N = np.stack([face.N(g) for g in face.gauss])
    dN = np.stack([face.dN(g) for g in face.gauss])
    xf = m.X[faces]
    area = np.linalg.norm(np.cross(np.einsum("fai,ga->fgi", xf, dN[:, :, 0]),
                                   np.einsum("fai,ga->fgi", xf, dN[:, :, 1])), axis=-1)
    f = np.zeros((m.n_nodes, 3))
    np.add.at(f[:, 1], faces, FORCE / (16.0 * T) * np.einsum("g,ga,fg->fa", face.weights, N, area))

    clamped = m.nodes[(0, -1)]
    fixed = np.unique(np.concatenate([(3 * clamped[:, None] + np.arange(3)).ravel(), 3 * np.arange(m.n_nodes) + 2]))
    bcs = [fem.BC(fixed, np.zeros(len(fixed)), f_dead=(k + 1) / steps * f.ravel()) for k in range(steps)]
    u, _, _ = sysm.run(states, bcs)
    tip = np.flatnonzero(np.hypot(m.X[:, 0] - 48.0, m.X[:, 1] - 60.0) < 1e-9)
    return float(np.asarray(u).reshape(-1, 3)[tip, 1].mean()), m.n_elem


def main():
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    log(f"Cook's membrane, mu {MU}, F {FORCE}, plane strain; tip vertical displacement v (mesh n x n x 1)")
    runs = {}
    for etype, element, ns in (("hex8", "hybrid", (2, 4, 8, 16, 32, 64)), ("hex8", "fbar", (2, 4, 8, 16, 32, 64)),
                               ("hex8", "standard", (2, 4, 8, 16, 32, 64)), ("tet10", "hybrid", (2, 4, 8, 16, 32)),
                               ("tet10", "standard", (2, 4, 8, 16, 32))):
        t0 = time.time()
        res = [tip_displacement(n, etype, element) for n in ns]
        runs[(etype, element)] = dict(zip(ns, (r[0] for r in res)))
        log(f"  {etype:5s} {element:8s} " + "  ".join(f"n={n}: {v:.5f}" for n, (v, _) in zip(ns, res))
            + f"   ({time.time() - t0:.0f}s)")

    v = runs[("hex8", "hybrid")]
    v4, v2, v1 = v[16], v[32], v[64]
    p = np.log2((v2 - v4) / (v1 - v2))
    ref = v1 + (v1 - v2) / (2 ** p - 1)
    log(f"\n  reference (hex8 hybrid, Richardson on n = 16/32/64): v = {ref:.5f}, observed order {p:.2f}")
    ok = True
    for (etype, element), vals in runs.items():
        n_fine = max(vals)
        err = abs(vals[n_fine] / ref - 1)
        expect_ok = element != "standard" or etype == "tet10"
        passed = err < 5e-3 if expect_ok else vals[2] / ref < 0.7
        ok &= passed
        note = "converged" if expect_ok else f"locks: n=2 gives {vals[2] / ref:.0%} of the reference"
        log(f"  {etype:5s} {element:8s} finest n={n_fine}: rel. diff to reference {err:.1e}   {note}  "
            f"{'PASS' if passed else 'FAIL'}")
    log("ALL PASS" if ok else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "cook.txt").write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    main()
