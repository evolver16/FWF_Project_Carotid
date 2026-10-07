"""Method of manufactured solutions: u_ex(X) (and q_ex(X) for the hybrid element) prescribed, body loaded with

    b = -Div P,   P = tau(F) F^-T - q J F^-T,   F = I + grad u_ex,   tau = material.sigma (Kirchhoff)

as consistent nodal forces f_a = int N_a b dV, nominal traction P N on the face x = 1, u = u_ex on the other faces;
relative errors under refinement h -> h/2

    e_L2 = |u - u_ex|_L2 / |u_ex|_L2,   e_H1 = |grad(u - u_ex)|_L2 / |grad u_ex|_L2,   e_q = |q - q_ex|_L2 / |q_ex|_L2

expected orders: hex8 (Q1) 2 / 1, tet10 (P2) 3 / 2; hybrid P0 pressure: q 1, u limited to 2 / 1.
Curved (non-affine) elements on a distorted cube, large strains, rotation, anisotropy (Fung fiber), F-bar,
hybrid with J_target = det F_ex (growth). Loads and errors integrated with Gauss rules of order 5 (tet: Duffy).
tet10 hybrid (P2/P0): element pressures oscillate between the 6 tets of a cube (not reduced by refinement), the
cube means converge; reported as "q L2 cubes". F-bar: localized near-singular mode under strong compression
(J ~ 0.65), hence a = 0.03 there.
Run: python -m verification.fem_mms
"""

import gc
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import fem
import materials
from fem import mesh as meshlib
from fem.pytree import pytree
from fem.tensor3 import det3, inv3

C10, K = 0.305, 6.1
OUT = pathlib.Path(__file__).parent / "results"


@pytree(("parts",))
class Sum:
    """tau = sum of the parts' tau (matrix + fibers)"""
    def __init__(self, parts):
        self.parts = parts

    def sigma(self, F):
        return sum(p.sigma(F) for p in self.parts)


@pytree(("material", "J_t"))
class Grown:
    def __init__(self, material, J_t):
        self.material, self.J_t = material, J_t


class Growth:
    """Elastic material with a prescribed J_target per Gauss point (hybrid element as with G&R)"""
    sigma_solver = staticmethod(jax.jit(lambda s, F: (s.material.sigma(F) / det3(F), None)))
    commit = staticmethod(lambda s, F, aux: s)
    J_target = staticmethod(lambda s: s.J_t)


def u_general(X, a=0.045):
    """stretches 0.78 ... 1.20, J >= 0.65; a = 0.06 (J >= 0.55) is past a bifurcation (negative tangent eigenvalue)"""
    k = jnp.array([[1.0, 0.7, 0.3], [0.4, 1.0, -0.6], [-0.5, 0.3, 1.0]])
    return a * jnp.sin(jnp.pi * k @ X + jnp.array([0.3, 1.1, -0.7]))


def u_isochoric(X, a=0.08, angle=0.2):
    """x = Q (X + w), grad w strictly lower triangular -> det F = 1"""
    w = a * jnp.array([0.0, jnp.sin(jnp.pi * X[0]), jnp.cos(jnp.pi * (X[0] + 0.8 * X[1]))])
    n = jnp.ones(3) / jnp.sqrt(3.0)
    W = jnp.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    Q = jnp.eye(3) + jnp.sin(angle) * W + (1 - jnp.cos(angle)) * W @ W
    return Q @ (X + w) - X


def u_fiber(X, M=jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)):
    """10 % stretch along the fiber plus u_general / 2: fiber taut everywhere (I4 > 1)"""
    return 0.1 * jnp.dot(M, X) * M + u_general(X, 0.03)


def q_field(X):
    return 0.05 * jnp.cos(jnp.pi * X[0]) * jnp.sin(jnp.pi * (X[1] + 0.5 * X[2])) + 0.02


def cube(n, etype):
    """unit cube, interior distorted by 0.04 sin(2 pi s_x) sin(2 pi s_y) sin(2 pi s_z) (1, 1, 1)"""
    return meshlib.structured((n, n, n), lambda s: s + 0.04 * np.prod(np.sin(2 * np.pi * s), axis=1, keepdims=True),
                              etype)


def rule(el, n=5):
    """Gauss-Legendre n^3 on [-1, 1]^3 (hex8) or collapsed onto the unit tetrahedron (tet10)"""
    g, w = np.polynomial.legendre.leggauss(n)
    if el.name == "hex8":
        pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        return pts, np.einsum("i,j,k->ijk", w, w, w).ravel()
    s, w = (g + 1) / 2, w / 2
    a, b, c = np.meshgrid(s, s, s, indexing="ij")
    wts = np.einsum("i,j,k->ijk", w, w, w) * (1 - a) ** 2 * (1 - b)
    return np.stack([a, b * (1 - a), c * (1 - a) * (1 - b)], -1).reshape(-1, 3), wts.ravel()


def face_rule(face, n=5):
    """Gauss-Legendre n^2 on [-1, 1]^2 (quad4) or collapsed onto the unit triangle (tri6)"""
    g, w = np.polynomial.legendre.leggauss(n)
    if face.name == "quad4":
        return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2), np.outer(w, w).ravel()
    s, w = (g + 1) / 2, w / 2
    a, b = np.meshgrid(s, s, indexing="ij")
    return np.stack([a, b * (1 - a)], -1).reshape(-1, 2), (np.outer(w, w) * (1 - a)).ravel()


def points(f, X):
    """f at every row of X, in batches (memory)"""
    return np.asarray(jax.jit(lambda X: jax.lax.map(f, X, batch_size=4096))(jnp.asarray(X)))


def solve_case(n, etype, element, material, u_ex, q_ex, model):
    """u = u_ex on five faces, exact nominal traction P N on x = 1 (pressure unique, volume free)"""
    m = cube(n, etype)
    el = m.element
    pts, wts = rule(el)
    N = np.stack([el.N(p) for p in pts])
    dN = np.stack([el.dN(p) for p in pts])
    J0 = np.einsum("eai,qaj->eqij", m.X[m.conn], dN)
    dNdX = np.einsum("qaj,eqji->eqai", dN, np.linalg.inv(J0))
    w = wts * np.linalg.det(J0)
    Xq = np.einsum("qa,eai->eqi", N, m.X[m.conn]).reshape(-1, 3)

    def P(X):
        F = jnp.eye(3) + jax.jacfwd(u_ex)(X)
        P = material.sigma(F) @ inv3(F).T
        return P if q_ex is None else P - q_ex(X) * det3(F) * inv3(F).T

    b = points(lambda X: -jnp.trace(jax.jacfwd(P)(X), axis1=1, axis2=2), Xq)
    f = np.zeros((m.n_nodes, 3))
    np.add.at(f, m.conn, np.einsum("eq,qa,eqi->eai", w, N, b.reshape(w.shape + (3,))))

    faces = m.faces[(0, 1)]
    face = el.face
    fpts, fwts = face_rule(face)
    Nf = np.stack([face.N(p) for p in fpts])
    dNf = np.stack([face.dN(p) for p in fpts])
    xf = m.X[faces]
    nA = np.cross(np.einsum("fai,qa->fqi", xf, dNf[:, :, 0]), np.einsum("fai,qa->fqi", xf, dNf[:, :, 1]))
    Pf = points(P, np.einsum("qa,fai->fqi", Nf, xf).reshape(-1, 3)).reshape(nA.shape + (3,))
    np.add.at(f, faces, np.einsum("q,qa,fqiJ,fqJ->fai", fwts, Nf, Pf, nA))

    sysm = fem.System(m, model, element=element)
    states = fem.broadcast_state(material, m.n_elem, sysm.n_gp)
    if model is Growth:
        Xg = fem.gauss_points(m.X, m.conn).reshape(-1, 3)
        J_t = points(lambda X: det3(jnp.eye(3) + jax.jacfwd(u_ex)(X)), Xg).reshape(m.n_elem, sysm.n_gp)
        states = Grown(states, J_t)
    boundary = np.unique(np.concatenate([m.nodes[k] for k in [(0, -1), (1, -1), (1, 1), (2, -1), (2, 1)]]))
    u_nodes = points(u_ex, m.X)
    bc = fem.BC((3 * boundary[:, None] + np.arange(3)).ravel(), u_nodes[boundary].ravel(), f_dead=f.ravel())
    u, it = sysm.solve(u_nodes.ravel(), states, bc)

    U = u.reshape(-1, 3)[m.conn]
    du = np.einsum("qa,eai->eqi", N, U).reshape(-1, 3) - points(u_ex, Xq)
    G_ex = points(jax.jacfwd(u_ex), Xq)
    dG = np.einsum("eai,eqaj->eqij", U, dNdX).reshape(-1, 3, 3) - G_ex
    wf = w.ravel()
    norm = lambda v: np.sqrt(np.sum(wf * np.sum(v.reshape(len(wf), -1) ** 2, axis=1)))
    errs = [norm(du) / norm(points(u_ex, Xq)), norm(dG) / norm(G_ex)]
    if q_ex is not None:
        qx = points(q_ex, Xq)
        errs.append(norm(np.repeat(sysm.last_p, len(wts)) - qx) / norm(qx))
        if etype == "tet10":
            vol = w.sum(1).reshape(-1, 6)
            q_cube = np.repeat((sysm.last_p.reshape(-1, 6) * vol).sum(1) / vol.sum(1), 6 * len(wts))
            errs.append(norm(q_cube - qx) / norm(qx))
    return errs, it, m.n_elem


def cases():
    comp, inc = materials.NeoHookean(C10, K), materials.NeoHookeanInc(C10)
    fiber = Sum([comp, materials.Fung(1.0, 2.0, np.array([1.0, 1.0, 0.0]))])
    stiff = materials.NeoHookean(C10, 2e4 * C10)
    hex_n, tet_n = (4, 8, 16), (2, 4, 8, 16)
    mild = lambda X: u_general(X, 0.03)
    yield "hex8  standard  compressible        ", "hex8", hex_n, "standard", comp, u_general, None, None, (2, 1)
    yield "tet10 standard  compressible        ", "tet10", tet_n, "standard", comp, u_general, None, None, (3, 2)
    yield "hex8  fbar      compressible, a=0.03", "hex8", hex_n, "fbar", comp, mild, None, None, (2, 1)
    yield "hex8  standard  fiber (Fung)        ", "hex8", hex_n, "standard", fiber, u_fiber, None, None, (2, 1)
    yield "tet10 standard  fiber (Fung)        ", "tet10", tet_n, "standard", fiber, u_fiber, None, None, (3, 2)
    yield "hex8  hybrid    incompressible, J=1 ", "hex8", hex_n, "hybrid", inc, u_isochoric, q_field, None, (2, 1, 1)
    yield "tet10 hybrid    incompressible, J=1 ", "tet10", tet_n, "hybrid", inc, u_isochoric, q_field, None, (2, 1, None, 1)
    yield "hex8  hybrid    J_target = det F_ex ", "hex8", hex_n, "hybrid", inc, u_general, q_field, Growth, (2, 1, 1)
    yield "tet10 hybrid    J_target = det F_ex ", "tet10", tet_n, "hybrid", inc, u_general, q_field, Growth, (2, 1, None, 1)
    yield "hex8  standard  K/C10 = 2e4 (locks) ", "hex8", hex_n, "standard", stiff, u_isochoric, None, None, None
    yield "hex8  fbar      K/C10 = 2e4         ", "hex8", hex_n, "fbar", stiff, u_isochoric, None, None, None


def main():
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    log(f"MMS on a distorted unit cube, neo-Hookean C10 {C10}, K {K}; errors relative, rate = log2(e_h / e_h/2)")
    ok = True
    for name, etype, ns, element, material, u_ex, q_ex, model, expected in cases():
        t0 = time.time()
        res = [solve_case(n, etype, element, material, u_ex, q_ex, model) for n in ns]
        errs = np.array([r[0] for r in res])
        rates = np.log2(errs[:-1] / errs[1:])
        jax.clear_caches()
        gc.collect()
        log(f"  {name} elements {' / '.join(str(r[2]) for r in res)}   Newton {' / '.join(str(r[1]) for r in res)}"
            f"   ({time.time() - t0:.0f}s)")
        for k, label in enumerate(("u L2       ", "u H1       ", "q L2       ", "q L2 cubes ")[:errs.shape[1]]):
            line = (f"      {label}  " + "  ".join(f"{e:.2e}" for e in errs[:, k])
                    + "   rates " + "  ".join(f"{r:.2f}" for r in rates[:, k]))
            if expected is not None and expected[k] is not None:
                passed = rates[-1, k] > expected[k] - 0.3
                ok &= passed
                line += f"   expected {expected[k]}  {'PASS' if passed else 'FAIL'}"
            log(line)
    log("ALL PASS" if ok else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "fem_mms.txt").write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    main()
