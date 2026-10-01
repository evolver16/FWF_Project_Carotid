"""Pressurized thick-walled tube (quarter cylinder, plane strain) vs closed-form solutions, neo-Hookean:

    small pressure, compressible     u_r = C1 r + C2 / r
                                     C1 = p a^2 / (2 (lam + mu)(b^2 - a^2)),  C2 = p a^2 b^2 / (2 mu (b^2 - a^2))
    finite pressure, incompressible  r^2 = R^2 + a^2 - A^2,  p = mu int_a^b (lam_t^2 - lam_t^-2) dr / r,  lam_t = r / R

1) hex8 convergence, small pressure      2) hex8 refinement at 15 kPa (nonlinear, no closed form)
3) element formulations at K/mu = 1e4    4) finite pressure, incompressible
5) hex8 vs tet10, small pressure         6) tet10 hybrid (P2/P0), finite pressure, incompressible
Run: python -m verification.fem_tube
"""

import pathlib
import time

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import fem
import materials
from fem import mesh as meshlib

R_I, T, L, C10, K = 5.0, 1.3, 0.22, 0.305, 6.1
OUT = pathlib.Path(__file__).parent / "results"


def cylinder_bc(m, p):
    """Symmetry u_y = 0 at theta = 0, u_x = 0 at theta = pi/2, u_z = 0 at both ends; pressure p inside."""
    fixed = np.unique(np.concatenate([3 * m.nodes[(1, -1)] + 1, 3 * m.nodes[(1, 1)] + 0,
                                      3 * m.nodes[(2, -1)] + 2, 3 * m.nodes[(2, 1)] + 2]))
    return fem.BC(fixed, np.zeros(len(fixed)), p=p)


def radial_u(m, u, side):
    """mean u . e_r over the inner (-1) or outer (+1) nodes"""
    idx = m.nodes[(0, side)]
    e_r = m.X[idx, :2] / np.linalg.norm(m.X[idx, :2], axis=1, keepdims=True)
    return jnp.mean(jnp.sum(jnp.asarray(u).reshape(-1, 3)[idx, :2] * e_r, axis=1))


def lame(p, a, b, mu, lam):
    """u_r(a) of the plane-strain Lame solution"""
    C1 = p * a ** 2 / (2 * (lam + mu) * (b ** 2 - a ** 2))
    C2 = p * a ** 2 * b ** 2 / (2 * mu * (b ** 2 - a ** 2))
    return C1 * a + C2 / a


def inner_radius_incompressible(p, A, B, mu):
    """Inner radius a of the incompressible tube under inner pressure p (finite strain)"""
    def p_of(a):
        b = np.sqrt(B ** 2 + a ** 2 - A ** 2)
        f = lambda r: (r ** 2 / (r ** 2 - a ** 2 + A ** 2) - (r ** 2 - a ** 2 + A ** 2) / r ** 2) / r
        return mu * quad(f, a, b, epsabs=1e-14, epsrel=1e-13)[0]

    return brentq(lambda a: p_of(a) - p, A, 3 * A, xtol=1e-14)


def solve(n, mat, p, element="standard", etype="hex8", steps=1):
    """u_r(a) of the quarter cylinder under inner pressure p, applied in steps"""
    m = meshlib.quarter_cylinder(R_I, T, L, n, etype)
    sysm = fem.System(m, pressure_faces=m.faces[(0, -1)], element=element)
    states = fem.broadcast_state(mat, m.n_elem, sysm.n_gp)
    if steps == 1:
        u, it = sysm.solve(np.zeros(sysm.n_dof), states, cylinder_bc(m, p))
    else:
        u, _, _ = sysm.run(states, [cylinder_bc(m, p * k / steps) for k in range(1, steps + 1)])
        it = None
    return float(radial_u(m, u, -1)), m, sysm, it


def main():
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    def errors(label, errs):
        log(f"  {label} rel err u_r(a): " + "  ".join(f"{e:.2e}" for e in errs)
            + "   ratios " + "  ".join(f"{errs[i] / errs[i + 1]:.1f}" for i in range(len(errs) - 1)))

    a, b, mu = R_I, R_I + T, 2 * C10
    comp = materials.NeoHookean(C10, K)
    hex_meshes = [(2, 12, 1), (4, 24, 1), (8, 48, 1), (16, 96, 1)]

    log(f"Pressurized tube r_i {R_I}, t {T}, L {L} (plane strain), neo-Hookean C10 {C10}")
    log(f"\n1) hex8, compressible K {K}, p = 1e-5 vs Lame")
    errs = [abs(solve(n, comp, 1e-5)[0] / lame(1e-5, a, b, mu, K - 2 * mu / 3) - 1) for n in hex_meshes]
    errors("hex8    ", errs)

    log("\n2) hex8, compressible, p = 15 kPa (nonlinear): u_r(a) under refinement")
    vals = []
    for n in hex_meshes + [(8, 60, 4)]:
        t0 = time.time()
        ur, m, _, it = solve(n, comp, 0.015)
        vals.append(ur)
        log(f"  n={n}  elements {m.n_elem:5d}  u_r(a) {ur:.8f} mm  Newton it {it}  ({time.time() - t0:.1f}s)")
    d = [abs(vals[i] - vals[i + 1]) for i in range(len(hex_meshes) - 1)]
    log("  successive differences " + "  ".join(f"{x:.2e}" for x in d)
        + "  ratios " + "  ".join(f"{d[i] / d[i + 1]:.1f}" for i in range(len(d) - 1)))

    ratio, meshes = 1e4, hex_meshes[:3]
    stiff, inc = materials.NeoHookean(C10, ratio * mu), materials.NeoHookeanInc(C10)
    variants = [("standard", stiff), ("fbar", stiff), ("hybrid", inc)]
    log(f"\n3) element formulations, p = 1e-5 vs Lame: compressible K/mu = {ratio:.0e} (standard, fbar), "
        "incompressible (hybrid)")
    for element, mat in variants:
        lam = None if element == "hybrid" else ratio * mu - 2 * mu / 3
        exact = (1e-5 * a ** 2 * b ** 2 / (2 * mu * (b ** 2 - a ** 2)) / a if lam is None
                 else lame(1e-5, a, b, mu, lam))
        errors(f"{element:8s}", [abs(solve(n, mat, 1e-5, element)[0] / exact - 1) for n in meshes])

    p = 0.1
    a_ex = inner_radius_incompressible(p, a, b, mu)
    log(f"\n4) finite pressure {p} MPa (p/mu = {p / mu:.2f}), exact incompressible a/A = {a_ex / a:.8f}")
    for element, mat in variants:
        errors(f"{element:8s}", [abs(solve(n, mat, p, element, steps=5)[0] / (a_ex - a) - 1) for n in meshes])

    tet_meshes = ((1, 6, 1), (2, 12, 1), (4, 24, 1))
    log("\n5) hex8 vs tet10, compressible, p = 1e-8 vs Lame")
    for etype in ("hex8", "tet10"):
        errors(f"{etype:8s}", [abs(solve(n, comp, 1e-8, etype=etype)[0] / lame(1e-8, a, b, mu, K - 2 * mu / 3) - 1)
                               for n in tet_meshes])

    log(f"\n6) tet10 hybrid (P2/P0), incompressible, p = {p} (a/A exact {a_ex / a:.6f})")
    errors("tet10   ", [abs(solve(n, inc, p, "hybrid", "tet10", steps=5)[0] / (a_ex - a) - 1) for n in tet_meshes])

    OUT.mkdir(exist_ok=True)
    (OUT / "fem_tube.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
