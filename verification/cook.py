"""Cook's membrane (plane strain, nearly / fully incompressible neo-Hookean): tapered panel (0,0)-(48,44)-(48,60)-(0,44)
clamped at x = 0, uniform dead shear traction tau = F / 16 on x = 48; vertical displacement of the tip (48, 60).
Bending with skewed elements: the standard locking benchmark. Reference: Richardson extrapolation of hex8 fbar,

    v_inf = v_h + (v_h - v_2h) / (2^p - 1),   p = log2((v_4h - v_2h) / (v_2h - v_h))

fbar, hybrid (hex8, tet10) and hybrid_p1 (tet10, P2/P1) must approach it (0.5 %); standard hex8 locks (n=2 below 70 %), standard tet10 locks
partially (slow convergence, within 5 % and improving). mu = 2 C10 = 80.194, K / mu = 5000 (fbar, standard)
or incompressible (hybrid), F = 32 per unit thickness, plane strain via u_z = 0.
hex8 hybrid (Q1/P0) is not inf-sup stable here (smallest / largest pressure Schur eigenvalue ~ h^2); it converges, but
the saddle-point LU needs PARDISO iterative refinement (without it, Newton failed from n = 32 on).
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
    material = materials.NeoHookeanInc(MU / 2) if element.startswith("hybrid") else materials.NeoHookean(MU / 2, RATIO * MU)
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
                               ("tet10", "hybrid_p1", (2, 4, 8, 16, 32)),
                               ("tet10", "standard", (2, 4, 8, 16, 32))):
        t0 = time.time()
        vals = {}
        for n in ns:
            try:
                vals[n] = tip_displacement(n, etype, element)[0]
            except RuntimeError:
                vals[n] = None
        runs[(etype, element)] = vals
        log(f"  {etype:5s} {element:8s} " + "  ".join(f"n={n}: " + ("failed " if v is None else f"{v:.5f}")
                                                    for n, v in vals.items()) + f"   ({time.time() - t0:.0f}s)")

    v = runs[("hex8", "fbar")]
    v4, v2, v1 = v[16], v[32], v[64]
    p = np.log2((v2 - v4) / (v1 - v2))
    ref = v1 + (v1 - v2) / (2 ** p - 1)
    log(f"\n  reference (hex8 fbar, Richardson on n = 16/32/64): v = {ref:.5f}, observed order {p:.2f}")
    ok = True
    for (etype, element), vals in runs.items():
        failed = [n for n, x in vals.items() if x is None]
        if failed:
            log(f"  {etype:5s} {element:8s} Newton failed for n = {failed}")
        ok &= not failed
        vals = {n: x for n, x in vals.items() if x is not None}
        ns = sorted(vals)
        err = abs(vals[ns[-1]] / ref - 1)
        if element != "standard":
            passed, note = err < 5e-3, "converged"
        elif etype == "tet10":
            passed = err < 5e-2 and err < abs(vals[ns[-2]] / ref - 1)
            note = "partial locking: slow convergence"
        else:
            passed, note = vals[2] / ref < 0.7, f"locks: n=2 gives {vals[2] / ref:.0%} of the reference"
        ok &= passed
        n_fine = ns[-1]
        log(f"  {etype:5s} {element:8s} finest n={n_fine}: rel. diff to reference {err:.1e}   {note}  "
            f"{'PASS' if passed else 'FAIL'}")
    log("ALL PASS" if ok else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "cook.txt").write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    main()
