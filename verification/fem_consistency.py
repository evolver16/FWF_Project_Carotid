"""Internal consistency of the FE code (no reference solution needed):

1) quadrature       sum w = V, int X dV = V X_c (distorted hex8, tet10)
2) tangent          assembled K = jacfwd of the global residual (exact AD), and vs central differences;
                    hex8 / tet10, standard / fbar / hybrid, matrix + fiber, follower pressure
3) Newton           e_k+1 ~ C e_k^2 (quadratic convergence of the full solver loop)
4) follower load    closed surface: f_p = -p dV/dx,  V = 1/3 oint x . n da;  sum f_p = 0,  sum x x f_p = 0,
                    load stiffness = p d^2V/dx^2 (symmetric)
5) invariance       slip (u . n = 0) = fixed dofs; rotated mesh + rotated slip normals: u' = Q u;
                    node / element renumbering; PARDISO = SuperLU
6) free growth      hybrid, J_target = c, symmetry planes only: u = (c^1/3 - 1) X, q = 0
7) adjoint          d(w . u)/d(C10, k1, p) by the adjoint vs central differences (slip supports, rotated mesh);
                    run() with and without checkpointing
Run: python -m verification.fem_consistency
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

import fem
import materials
from fem import mesh as meshlib
from fem.core import LinearSolver
from verification.fem_mms import Growth, Grown, Sum, cube
from verification.fem_tube import cylinder_bc

C10, K, K1, K2 = 0.305, 6.1, 1.0, 2.0
OUT = pathlib.Path(__file__).parent / "results"


def rotation(axis, angle):
    n = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    W = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(angle) * W + (1 - np.cos(angle)) * W @ W


def material_for(element, M=(0.0, 0.0, 1.0)):
    matrix = materials.NeoHookeanInc(C10) if element == "hybrid" else materials.NeoHookean(C10, K)
    return Sum([matrix, materials.Fung(K1, K2, np.asarray(M))])


def tube(etype, Q=np.eye(3)):
    """quarter cylinder (plane strain) rotated by Q, slip supports (rotated symmetry planes and ends)"""
    m = meshlib.quarter_cylinder(5.0, 1.3, 0.5, (2, 12, 1), etype)
    m.X = m.X @ Q.T
    sets = [((1, -1), [0, 1, 0]), ((1, 1), [1, 0, 0]), ((2, -1), [0, 0, 1]), ((2, 1), [0, 0, 1])]
    nodes = np.concatenate([m.nodes[k] for k, _ in sets])
    normals = np.concatenate([np.tile(Q @ np.array(n, dtype=float), (len(m.nodes[k]), 1)) for k, n in sets])
    return m, (nodes, normals)


def main():
    lines = []
    ok = True

    def log(text):
        print(text, flush=True)
        lines.append(text)

    def check(label, err, tol):
        nonlocal ok
        ok &= bool(err < tol)
        log(f"  {label:58s} {err:.1e}  {'PASS' if err < tol else 'FAIL'}")

    rng = np.random.default_rng(0)

    log("1) quadrature: sum w = V, int X dV = V X_c")
    for name, m in (("hex8 distorted cube", cube(4, "hex8")), ("tet10 box", meshlib.box((2, 2, 2), (1, 1, 1), etype="tet10"))):
        _, w = fem.geometry(m.X, m.conn)
        X = fem.gauss_points(m.X, m.conn)
        check(f"{name}: |sum w - 1|, |int X dV - 1/2|",
              max(abs(w.sum() - 1), np.abs(np.einsum("eg,egi->i", w, X) - 0.5).max()), 1e-13)

    log("\n2) tangent: assembled K vs jacfwd of the residual (exact) and central differences, deformed state")
    for etype in ("hex8", "tet10"):
        m = meshlib.box((2, 2, 2), (1, 1, 1), perturb=0.2 if etype == "hex8" else 0.0, etype=etype)
        for element in ("standard", "fbar", "hybrid"):
            sysm = fem.System(m, pressure_faces=m.faces[(2, 1)], element=element)
            states = fem.broadcast_state(material_for(element, (1.0, 1.0, 0.5)), m.n_elem, sysm.n_gp)
            x = np.concatenate([0.05 * rng.standard_normal(sysm.n_dof) + 0.1 * m.X.ravel(),
                                0.05 * rng.standard_normal(sysm.n_p)])
            f0, p = jnp.zeros(sysm.n_dof), 0.03
            r = lambda x: sysm.residual(x, states, p, f0)
            K_fe = sysm._K_free(x, states, p, np.ones(sysm.n_x, dtype=bool)).toarray()
            K_ad = np.asarray(jax.jacfwd(r)(jnp.asarray(x)))
            v = rng.standard_normal(sysm.n_x)
            h = 1e-6
            fd = (np.asarray(r(jnp.asarray(x + h * v))) - np.asarray(r(jnp.asarray(x - h * v)))) / (2 * h)
            scale = np.abs(K_ad).max()
            check(f"{etype:5s} {element:8s} |K - dr/dx|_max / |K|_max", np.abs(K_fe - K_ad).max() / scale, 1e-12)
            check(f"{etype:5s} {element:8s} |K v - FD|_max / |K v|_max",
                  np.abs(K_fe @ v - fd).max() / np.abs(K_fe @ v).max(), 1e-7)

    log("\n3) Newton convergence order, tube p = 0.1 MPa from u = 0 (hybrid, matrix + axial fibers)")
    for etype in ("hex8", "tet10"):
        m = meshlib.quarter_cylinder(5.0, 1.3, 0.5, (2, 12, 1), etype)
        sysm = fem.System(m, pressure_faces=m.faces[(0, -1)], element="hybrid")
        states = fem.broadcast_state(material_for("hybrid"), m.n_elem, sysm.n_gp)
        bc = cylinder_bc(m, 0.1)
        cons = sysm._constraints(bc)
        x, f0, errs = cons.project(np.zeros(sysm.n_x)), jnp.zeros(sysm.n_dof), []
        for _ in range(12):
            r = cons.restrict(np.asarray(sysm.residual(jnp.asarray(x), states, bc.p, f0)))
            errs.append(np.abs(r).max())
            if errs[-1] < 1e-13:
                break
            x = x - cons.prolong(spla.spsolve(sysm._K_red(x, states, bc.p, cons), r))
        e = np.log(np.array(errs))
        order = [(e[k + 1] - e[k]) / (e[k] - e[k - 1]) for k in range(1, len(e) - 1) if errs[k + 1] > 1e-11]
        log(f"  {etype}: |r| " + " ".join(f"{v:.1e}" for v in errs))
        check(f"{etype} 2 - max observed order log(e_k+1/e_k)/log(e_k/e_k-1)", 2 - max(order), 0.2)

    log("\n4) follower pressure on a closed surface (all six faces of a deformed box)")
    for etype in ("hex8", "tet10"):
        m = meshlib.box((2, 2, 2), (1, 1, 1), perturb=0.2 if etype == "hex8" else 0.0, etype=etype)
        faces = np.concatenate([m.faces[(a, s)] for a in range(3) for s in (-1, 1)])
        sysm = fem.System(m, pressure_faces=faces)
        states = fem.broadcast_state(materials.NeoHookean(C10, K), m.n_elem, sysm.n_gp)
        u = jnp.asarray(0.05 * rng.standard_normal(sysm.n_dof) + 0.1 * np.sin(3 * m.X).ravel())
        p, f0 = 0.07, jnp.zeros(sysm.n_dof)
        f_p = np.asarray(sysm.residual(u, states, 0.0, f0) - sysm.residual(u, states, p, f0)).reshape(-1, 3)

        def volume(u):
            x = (sysm.X + u.reshape(-1, 3))[sysm.faces]
            a1 = jnp.einsum("fai,ga->fgi", x, sysm.dN2[:, :, 0])
            a2 = jnp.einsum("fai,ga->fgi", x, sysm.dN2[:, :, 1])
            xg = jnp.einsum("fai,ga->fgi", x, sysm.N2)
            return jnp.einsum("g,fgi,fgi->", sysm.w2, xg, jnp.cross(a1, a2)) / 3

        dV = np.asarray(jax.grad(volume)(u)).reshape(-1, 3)
        x = m.X + np.asarray(u).reshape(-1, 3)
        check(f"{etype:5s} |f_p + p dV/dx|_max / |f_p|_max", np.abs(f_p + p * dV).max() / np.abs(f_p).max(), 1e-12)
        check(f"{etype:5s} |sum f_p|, |sum x x f_p| / (|f_p|_max L)",
              max(np.abs(f_p.sum(0)).max(), np.abs(np.cross(x, f_p).sum(0)).max()) / np.abs(f_p).max(), 1e-12)
        free = np.ones(sysm.n_x, dtype=bool)
        K_p = (sysm._K_free(np.asarray(u), states, p, free) - sysm._K_free(np.asarray(u), states, 0.0, free)).toarray()
        H = p * np.asarray(jax.hessian(volume)(u))
        check(f"{etype:5s} |K_p - p d2V/dx2|_max / |K_p|_max", np.abs(K_p - H).max() / np.abs(K_p).max(), 1e-12)
        check(f"{etype:5s} |K_p - K_p^T|_max / |K_p|_max", np.abs(K_p - K_p.T).max() / np.abs(K_p).max(), 1e-12)

    log("\n5) invariance: slip = fixed, rotated mesh, renumbering, linear solver backend (tube, p = 0.05 MPa)")
    Q = rotation([1.0, -2.0, 0.5], 0.7)
    for etype, element in (("hex8", "hybrid"), ("hex8", "standard"), ("tet10", "hybrid"), ("tet10", "fbar")):
        m, _ = tube(etype)
        sysm = fem.System(m, pressure_faces=m.faces[(0, -1)], element=element)
        states = fem.broadcast_state(material_for(element), m.n_elem, sysm.n_gp)
        u_ref, _ = sysm.solve(np.zeros(sysm.n_dof), states, cylinder_bc(m, 0.05))
        p_ref = sysm.last_p.copy()
        scale = np.abs(u_ref).max()
        tag = f"{etype:5s} {element:8s}"

        _, slip = tube(etype)
        u, _ = sysm.solve(np.zeros(sysm.n_dof), states, fem.BC(np.zeros(0, int), np.zeros(0), p=0.05, slip=slip))
        check(f"{tag} slip vs fixed dofs  max|du| / max|u|", np.abs(u - u_ref).max() / scale, 1e-10)

        mq, slip = tube(etype, Q)
        sq = fem.System(mq, pressure_faces=mq.faces[(0, -1)], element=element)
        st = fem.broadcast_state(material_for(element, Q @ np.array([0.0, 0.0, 1.0])), mq.n_elem, sq.n_gp)
        u, _ = sq.solve(np.zeros(sq.n_dof), st, fem.BC(np.zeros(0, int), np.zeros(0), p=0.05, slip=slip))
        err = max(np.abs(u.reshape(-1, 3) - u_ref.reshape(-1, 3) @ Q.T).max() / scale,
                  np.abs(sq.last_p - p_ref).max() / max(np.abs(p_ref).max(), 1e-300) if sq.n_p else 0.0)
        check(f"{tag} rotated mesh: max|u' - Q u|, max|q' - q| (rel)", err, 1e-10)

        pn, pe = rng.permutation(m.n_nodes), rng.permutation(m.n_elem)
        X = np.empty_like(m.X)
        X[pn] = m.X
        mr = meshlib.Mesh(X=X, conn=pn[m.conn[pe]], nodes={k: pn[v] for k, v in m.nodes.items()},
                          faces={k: pn[v] for k, v in m.faces.items()})
        sr = fem.System(mr, pressure_faces=mr.faces[(0, -1)], element=element)
        u, _ = sr.solve(np.zeros(sr.n_dof), fem.broadcast_state(material_for(element), mr.n_elem, sr.n_gp),
                        cylinder_bc(mr, 0.05))
        err = max(np.abs(u.reshape(-1, 3)[pn] - u_ref.reshape(-1, 3)).max() / scale,
                  np.abs(sr.last_p - p_ref[pe]).max() / np.abs(p_ref).max() if sr.n_p else 0.0)
        check(f"{tag} renumbered nodes + elements: max|du|, max|dq| (rel)", err, 1e-10)

        sysm.linear = LinearSolver(backend="superlu" if sysm.linear.backend == "pardiso" else "pardiso")
        u, _ = sysm.solve(np.zeros(sysm.n_dof), states, cylinder_bc(m, 0.05))
        check(f"{tag} {sysm.linear.backend} vs default backend  max|du| / max|u|", np.abs(u - u_ref).max() / scale, 1e-10)

    log("\n6) free growth, hybrid, J_target = 1.3, symmetry planes: u = (1.3^1/3 - 1) X, q = 0")
    for etype in ("hex8", "tet10"):
        m = cube(2, etype) if etype == "hex8" else meshlib.box((2, 2, 2), (1, 1, 1), etype="tet10")
        sysm = fem.System(m, Growth, element="hybrid")
        states = Grown(fem.broadcast_state(materials.NeoHookeanInc(C10), m.n_elem, sysm.n_gp),
                       jnp.full((m.n_elem, sysm.n_gp), 1.3))
        fixed = np.concatenate([3 * m.nodes[(a, -1)] + a for a in range(3)])
        u, _ = sysm.solve(np.zeros(sysm.n_dof), states, fem.BC(fixed, np.zeros(len(fixed))))
        err = max(np.abs(u - (1.3 ** (1 / 3) - 1) * m.X.ravel()).max(), np.abs(sysm.last_p).max())
        check(f"{etype:5s} max|u - u_ex|, max|q|", err, 1e-10)

    log("\n7) adjoint vs central differences: d(w . u)/d(C10, k1, p), rotated tube with slip supports")
    for etype, element in (("hex8", "hybrid"), ("tet10", "standard"), ("hex8", "fbar")):
        m, slip = tube(etype, Q)
        sysm = fem.System(m, pressure_faces=m.faces[(0, -1)], element=element)
        w = jnp.asarray(rng.standard_normal(sysm.n_dof))
        M = Q @ np.array([0.0, 0.0, 1.0])

        def objective(par, checkpoint=None, steps=1):
            matrix = (materials.NeoHookeanInc(par["C10"]) if element == "hybrid"
                      else materials.NeoHookean(par["C10"], K))
            states = fem.broadcast_state(Sum([matrix, materials.Fung(par["k1"], K2, M)]), m.n_elem, sysm.n_gp)
            bcs = [fem.BC(np.zeros(0, int), np.zeros(0), p=par["p"] * (k + 1) / steps, slip=slip) for k in range(steps)]
            u, _, _ = sysm.run(states, bcs, tol=1e-12, checkpoint=checkpoint)
            return jnp.dot(w, u)

        par = {k: jnp.asarray(v) for k, v in dict(C10=C10, k1=K1, p=0.05).items()}
        g = jax.grad(objective)(par)
        for k in par:
            h = 1e-5 * float(par[k])
            fd = (objective({**par, k: par[k] + h}) - objective({**par, k: par[k] - h})) / (2 * h)
            check(f"{etype:5s} {element:8s} d/d{k:4s} |adjoint - FD| / |FD|", abs(float(g[k] - fd)) / abs(float(fd)), 1e-6)
        g1 = jax.grad(lambda p: objective(p, None, 4))(par)
        g2 = jax.grad(lambda p: objective(p, 2, 4))(par)
        check(f"{etype:5s} {element:8s} 4 steps: checkpoint=2 vs none (rel)",
              max(abs(float(g1[k] - g2[k])) / abs(float(g1[k])) for k in par), 1e-10)

    log("ALL PASS" if ok else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "fem_consistency.txt").write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    main()
