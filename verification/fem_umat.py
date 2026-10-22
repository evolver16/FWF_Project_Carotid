"""Abaqus UMAT interface (fem.umat) with the compressible neo-Hooke UMAT of verification/umat/neohooke.for against the
Python material (materials.NeoHookean, same W = C10 (I1bar - 3) + K/2 (J - 1)^2):

1) material point   Cauchy stress, random F
2) tangent          d sigma/dF from DDSDDE (Jaumann rate of Kirchhoff / J -> sigma_dot) vs exact AD of the Python
                    material and vs central differences of the UMAT stress
3) FE               distorted 2x2x2 hex8 box, x = 0 clamped, x = L displaced (stretch + shear, inhomogeneous):
                    u and Newton iterations of UMAT vs Python material; standard / fbar / hybrid
4) increments       3 load steps: STATEV(2) = committed increments, STATEV(3) = TIME(2) + DTIME, STATEV(1) = det F,
                    SDVINI: STATEV(4) = x of the Gauss point
5) UHYPER           compressible Mooney-Rivlin UHYPER (verification/umat/mooney_rivlin.for): stress and tangent vs the
                    same energy written in JAX; c2 = 0: FE vs materials.NeoHookean (C10 = c1/2, K = 2/D1)
Run: python -m verification.fem_umat
"""

import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import fem
import materials
from fem import mesh as meshlib
from fem.tensor3 import det3
from fem.umat import Uhyper, Umat

C10, K, L = 0.305, 6.1, 1.0
OUT = pathlib.Path(__file__).parent / "results"
SRC = pathlib.Path(__file__).parent / "umat" / "neohooke.for"
MR = pathlib.Path(__file__).parent / "umat" / "mooney_rivlin.for"


def main(tol=1e-8):
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    ok = True
    umat = Umat([SRC], nstatv=4, sdvini=True)
    py = materials.NeoHookean(C10, K)
    cauchy = lambda F: py.sigma(F) / det3(F)
    m = meshlib.box((2, 2, 2), (L, L, L), perturb=0.2)
    st = umat.state(m, [C10, K], dtime=0.5)
    s0 = jax.tree.map(lambda a: a[0, 0], st)
    rng = np.random.default_rng(0)
    Fs = [np.eye(3) + 0.3 * rng.standard_normal((3, 3)) for _ in range(5)]
    Fs = [F if np.linalg.det(F) > 0.3 else np.eye(3) for F in Fs]

    log(f"UMAT interface, compressible neo-Hooke UMAT ({SRC.name}) vs materials.NeoHookean, C10 {C10}, K {K}")
    log("1) material point, Cauchy stress")
    e = max(float(jnp.abs(umat.sigma_solver(s0, jnp.asarray(F))[0] - cauchy(jnp.asarray(F))).max()
                  / jnp.abs(cauchy(jnp.asarray(F))).max()) for F in Fs)
    ok &= e < tol
    log(f"  max rel error over {len(Fs)} random F: {e:.1e}  {'PASS' if e < tol else 'FAIL'}")

    log("2) tangent d sigma / dF")
    e_ad, e_fd = 0.0, 0.0
    for F in Fs:
        F = jnp.asarray(F)
        A = jax.jacfwd(lambda f: umat.sigma_solver(s0, f)[0])(F)
        A_py = jax.jacfwd(cauchy)(F)
        h = 1e-6
        A_fd = np.zeros((3, 3, 3, 3))
        for k in range(3):
            for l in range(3):
                dF = np.zeros((3, 3))
                dF[k, l] = h
                A_fd[..., k, l] = (np.asarray(umat.sigma_solver(s0, F + dF)[0])
                                   - np.asarray(umat.sigma_solver(s0, F - dF)[0])) / (2 * h)
        e_ad = max(e_ad, float(jnp.abs(A - A_py).max() / jnp.abs(A_py).max()))
        e_fd = max(e_fd, float(np.abs(np.asarray(A) - A_fd).max() / np.abs(A_fd).max()))
    ok &= e_ad < tol and e_fd < 1e-6
    log(f"  vs exact AD of the Python material {e_ad:.1e}, vs central differences of the UMAT stress {e_fd:.1e}  "
        f"{'PASS' if e_ad < tol and e_fd < 1e-6 else 'FAIL'}")

    log("3) FE, distorted 2x2x2 hex8 box, x = 0 clamped, x = L: u = (0.3, 0.1, -0.05)")
    x0, x1 = m.nodes[(0, -1)], m.nodes[(0, 1)]
    dofs = np.concatenate([3 * x0[:, None] + np.arange(3), 3 * x1[:, None] + np.arange(3)]).ravel()
    vals = np.concatenate([np.zeros((len(x0), 3)), np.tile([0.3, 0.1, -0.05], (len(x1), 1))]).ravel()
    order = np.argsort(dofs)
    bc = fem.BC(dofs[order], vals[order])
    for element in ("standard", "fbar", "hybrid"):
        res = {}
        for name, model, states in (("python", None, fem.broadcast_state(py, m.n_elem, 8)), ("umat", umat, st)):
            sysm = fem.System(m, model, element=element)
            t0 = time.time()
            u, it = sysm.solve(np.zeros(sysm.n_dof), states, bc)
            res[name] = (np.asarray(u), it, time.time() - t0)
        e = np.abs(res["umat"][0] - res["python"][0]).max() / np.abs(res["python"][0]).max()
        same = e < tol and res["umat"][1] == res["python"][1]
        ok &= same
        log(f"  {element:8s}  max rel |u_umat - u_py| {e:.1e}  Newton {res['umat'][1]} vs {res['python'][1]}  "
            f"time {res['umat'][2]:.1f} s vs {res['python'][2]:.1f} s (incl. compilation)  {'PASS' if same else 'FAIL'}")

    log("4) increments: 3 load steps (DTIME 0.5), STATEV after each commit")
    sysm = fem.System(m, umat)
    bcs = [fem.BC(bc.fixed, bc.values * f) for f in (1 / 3, 2 / 3, 1.0)]
    u, states, _ = sysm.run(st, bcs)
    sv = np.asarray(states.statev)
    J = np.asarray(det3(sysm.gauss_F(u)))
    xg = np.asarray(fem.gauss_points(m.X, m.conn))
    checks = {"STATEV(2) = 3 increments": np.abs(sv[..., 1] - 3).max(),
              "STATEV(3) = TIME(2) + DTIME = 1.5": np.abs(sv[..., 2] - 1.5).max(),
              "STATEV(1) = det F": np.abs(sv[..., 0] - J).max(),
              "STATEV(4) = x (SDVINI)": np.abs(sv[..., 3] - xg[..., 0]).max(),
              "KINC = 4, TIME = (1.5, 1.5)": max(np.abs(np.asarray(states.kinc) - 4).max(),
                                                 np.abs(np.asarray(states.time) - 1.5).max())}
    for name, err in checks.items():
        ok &= err < tol
        log(f"  {name:36s} max err {err:.1e}  {'PASS' if err < tol else 'FAIL'}")

    log("5) UHYPER, compressible Mooney-Rivlin (D1, c1, c2) = (2/K, 2 C10, 0.4)")
    uh = Uhyper([MR])
    D1, c1, c2 = 2 / K, 2 * C10, 0.4
    s_mr = jax.tree.map(lambda a: a[0, 0], uh.state(m, [D1, c1, c2]))

    def mr_jax(F):
        J = det3(F)
        Bb = J ** (-2 / 3) * F @ F.T
        I1 = jnp.trace(Bb)
        I2 = 0.5 * (I1 ** 2 - jnp.trace(Bb @ Bb))
        return c1 / 2 * (I1 - 3) + c2 / 2 * (I2 - 3) + (J - 1) ** 2 / D1

    ref = lambda F: jax.grad(mr_jax)(F) @ F.T / det3(F)
    e_s = max(float(jnp.abs(uh.sigma_solver(s_mr, jnp.asarray(F))[0] - ref(jnp.asarray(F))).max()
                    / jnp.abs(ref(jnp.asarray(F))).max()) for F in Fs)
    e_t = max(float(jnp.abs(jax.jacfwd(lambda f: uh.sigma_solver(s_mr, f)[0])(jnp.asarray(F))
                            - jax.jacfwd(ref)(jnp.asarray(F))).max() / jnp.abs(jax.jacfwd(ref)(jnp.asarray(F))).max())
              for F in Fs)
    ok &= e_s < tol and e_t < tol
    log(f"  stress {e_s:.1e}, tangent {e_t:.1e} vs the energy in JAX  {'PASS' if e_s < tol and e_t < tol else 'FAIL'}")
    for element in ("standard", "hybrid"):
        u_py, it_py = fem.System(m, element=element).solve(np.zeros(3 * m.n_nodes), fem.broadcast_state(py, m.n_elem, 8),
                                                           bc)
        sysm = fem.System(m, uh, element=element)
        u_uh, it_uh = sysm.solve(np.zeros(sysm.n_dof), uh.state(m, [D1, c1, 0.0]), bc)
        e = np.abs(np.asarray(u_uh) - np.asarray(u_py)).max() / np.abs(np.asarray(u_py)).max()
        same = e < tol and it_uh == it_py
        ok &= same
        log(f"  FE {element:8s} c2 = 0 vs NeoHookean: max rel |du| {e:.1e}  Newton {it_uh} vs {it_py}  "
            f"{'PASS' if same else 'FAIL'}")
    log("ALL PASS" if ok else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "fem_umat.txt").write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    main()
