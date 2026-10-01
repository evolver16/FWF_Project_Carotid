"""G&R inside the FE code: HCMM patch test, single elements vs the material point (standard and hybrid), adjoint
through the hybrid element; artery G&R per element formulation (standard, F-bar, hybrid, tet10) and the tet10 adjoint
with inclined supports.  The pure FE checks are in verification/ (fem_analytic, fem_tube, mesh_io, buckling)."""

import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import fem
import hcmm
import setups
from fem import mesh as meshlib
from setups import MODELS, TARGETS, VARIANTS
from studies.artery import artery_mesh, artery_par, artery_simulate
from studies.common import Log, outputs
from verification import material_point

DRIVEN = material_point.DRIVEN_AXIS
L_BOX = 50.0


def box_bc(m, case):
    """Maes Fig. 1 cases on one element: symmetry planes, U stretch / S traction / F force on the driven face"""
    top = m.nodes[(DRIVEN, 1)]
    fixed = np.concatenate([3 * m.nodes[(0, -1)] + 0, 3 * m.nodes[(0, 1)] + 0,
                            3 * m.nodes[(DRIVEN, -1)] + DRIVEN, 3 * m.nodes[(2, -1)] + 2])
    target = TARGETS[case]
    if case == "U":
        vals = np.concatenate([np.zeros(len(fixed)), np.full(len(top), (target - 1) * L_BOX)])
        return fem.BC(np.concatenate([fixed, 3 * top + DRIVEN]), vals), None
    if case == "S":
        return fem.BC(fixed, np.zeros(len(fixed)), p=-target), m.faces[(DRIVEN, 1)]
    f = np.zeros(3 * m.n_nodes)
    f[3 * top + DRIVEN] = target / len(top)
    return fem.BC(fixed, np.zeros(len(fixed)), f_dead=f), None


# ---------- single elements vs the material point, adjoint through the hybrid element ----------

def fe_vs_material_point():
    log = Log()
    build, module = VARIANTS["HCMM dep"]

    log("1) HCMM patch test (model E): 2x2x2 distorted box, u = (F0 - I) X on the boundary")
    F0 = jnp.array([[1.08, 0.05, -0.02], [0.03, 0.95, 0.04], [-0.01, 0.02, 1.03]])
    m = meshlib.box((2, 2, 2), (1.0, 1.0, 1.0), perturb=0.2)
    boundary = np.unique(np.concatenate(list(m.nodes.values())))
    fixed = (3 * boundary[:, None] + np.arange(3)).ravel()
    u_exact = ((np.asarray(F0) - np.eye(3)) @ m.X.T).T.ravel()
    state = setups.build_hcmm(setups.maes_specs("E", "S"), 10.0, MODELS["E"]["ag"])
    sysm = fem.System(m, hcmm)
    states = fem.broadcast_state(state, m.n_elem)
    u, it = sysm.solve(np.zeros(sysm.n_dof), states, fem.BC(fixed, u_exact[fixed]))
    sig0 = hcmm.sigma_solver(state, F0)[0]
    eu = float(np.abs(u - u_exact).max())
    es = float(jnp.abs(sysm.stress(jnp.asarray(u), states) - sig0).max() / jnp.abs(sig0).max())
    log(f"  Newton it {it}  max|u - u_exact| {eu:.1e}  max rel sigma err {es:.1e}  "
        f"{'PASS' if eu < 1e-10 and es < 1e-10 else 'FAIL'}")

    m = meshlib.box((1, 1, 1), (L_BOX, L_BOX, L_BOX))
    top = m.nodes[(DRIVEN, 1)]
    for title, element, models in (("2) one hex8", "standard", ("B", "E")),
                                   ("3) one hybrid hex8, incompressible", "hybrid", ("A", "D"))):
        log(f"\n{title} with the Fig. 1 BCs vs material_point.py (HCMM dep, 100 steps of 10 days)")
        for model_name in models:
            for case in ("U", "S", "F"):
                specs, ag = setups.maes_specs(model_name, case), MODELS[model_name]["ag"]
                ref = material_point.run(build(specs, 10.0, ag), module, setups.bc_for(model_name, case), n_steps=100)
                bc, faces = box_bc(m, case)
                sysm = fem.System(m, module, pressure_faces=faces, element=element)

                def on_step(u, st):
                    s = float(sysm.stress(jnp.asarray(u), st)[0, 0, DRIVEN, DRIVEN])
                    return 1 + u[3 * top[0] + DRIVEN] / L_BOX, s - float(sysm.last_p[0] if sysm.n_p else 0.0)

                t0 = time.time()
                _, _, hist = sysm.run(fem.broadcast_state(build(specs, 10.0, ag), m.n_elem), [bc] * 100, on_step=on_step)
                key, q = ("sigma_driven", 1) if case == "U" else ("lam_driven", 0)
                fe_q = np.array([h[q] for h in hist])
                e = float(np.abs(fe_q - np.asarray(ref[key])).max())
                log(f"  {model_name} {case}  {key}(1000 d) FE {fe_q[-1]:.6f}  1-point {float(ref[key][-1]):.6f}  "
                    f"max diff {e:.1e}  ({time.time() - t0:.1f}s)  {'PASS' if e < 1e-8 else 'FAIL'}")

    log("\n4) adjoint through the hybrid element: model D case S, 20 steps, d lam_driven(end) / d (C10_matrix, p)")
    st0 = fem.broadcast_state(build(setups.maes_specs("D", "S"), 10.0, None), m.n_elem)
    bc, faces = box_bc(m, "S")
    sysm = fem.System(m, module, pressure_faces=faces, element="hybrid")

    def lam_end(C10, p):
        c0 = st0.constituents[0]
        c0 = c0.replace(material=type(c0.material)(jnp.broadcast_to(C10, c0.material.C10.shape)))
        st = st0.replace(constituents=[c0] + st0.constituents[1:])
        u, _, _ = sysm.run(st, [replace(bc, p=p)] * 20, tol=1e-12)
        return 1 + u[3 * top[0] + DRIVEN] / L_BOX

    x0 = (jnp.asarray(MODELS["D"]["matrix"]["C10"]), jnp.asarray(-TARGETS["S"]))
    g = jax.grad(lam_end, argnums=(0, 1))(*x0)
    for i, name in enumerate(("C10", "p")):
        h = 1e-6 * abs(float(x0[i]))
        xp, xm = list(x0), list(x0)
        xp[i], xm[i] = x0[i] + h, x0[i] - h
        fd = (float(lam_end(*xp)) - float(lam_end(*xm))) / (2 * h)
        err = abs(float(g[i]) - fd) / abs(fd)
        log(f"  {name:4s} adjoint {float(g[i]): .8e}   FD {fd: .8e}   rel err {err:.1e}  "
            f"{'PASS' if err < 1e-5 else 'FAIL'}")

    log.save(outputs("fe_gr/fe_vs_material_point", "fe_vs_material_point.txt")[0])


# ---------- artery G&R per element formulation, tet10 adjoint with inclined supports ----------

def element_formulations(steps=100):
    log = Log()
    log(f"1) artery G&R (model E, {steps} steps), elastin K/C10 as given or incompressible:"
        f" lambda_theta(day {10 * steps}), max spread of u_r(a) over theta")
    C10_e = float(artery_par()["C10"])
    variants = [("standard", "compressible", 20.0), ("standard", "compressible", 1e3),
                ("fbar", "compressible", 1e2), ("fbar", "compressible", 1e3), ("hybrid", "incompressible", None)]
    for n, subset in (((2, 15, 1), variants), ((8, 60, 4), [variants[i] for i in (0, 2, 4)])):
        for element, elastin, K in subset:
            m, sysm, Qloc = artery_mesh(n, element)
            par = artery_par() if K is None else artery_par(K=K * C10_e)
            idx = m.nodes[(0, -1)]
            e_r = m.X[idx, :2] / np.linalg.norm(m.X[idx, :2], axis=1, keepdims=True)
            a0 = float(np.linalg.norm(m.X[idx[0], :2]))

            def measure(u):
                ur = (u.reshape(-1, 3)[idx, :2] * e_r).sum(1)
                return 1 + ur.mean() / a0, ur.max() - ur.min()

            label = f"  n={n}  {element:8s} K/C10 {'inc' if K is None else f'{K:g}':5s}"
            t0 = time.time()
            try:
                lam_t, spread = artery_simulate(m, sysm, Qloc, par, n_steps=steps, elastin=elastin, measure=measure)
                log(f"{label} lambda_theta {float(lam_t[-1]):.6f}  spread {float(spread.max()):.1e} mm  "
                    f"({time.time() - t0:.1f}s)")
            except (RuntimeError, np.linalg.LinAlgError) as e:
                log(f"{label} unstable: {e}  ({time.time() - t0:.1f}s)")

    log(f"\n2) artery G&R (model E, {steps} steps), hex8 vs tet10: lambda_theta(day {10 * steps})")
    for etype, n in (("hex8", (8, 60, 4)), ("tet10", (2, 15, 1)), ("tet10", (4, 30, 1))):
        m = meshlib.quarter_cylinder(n=n, etype=etype)
        sysm = fem.System(m, hcmm, pressure_faces=m.faces[(0, -1)])
        Q = jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))
        t0 = time.time()
        lam_t = artery_simulate(m, sysm, Q, artery_par(), n_steps=steps)[0]
        log(f"  {etype:6s} n={n}  {m.n_elem:5d} elements  {sysm.n_dof:6d} dofs  lambda_theta {float(lam_t[-1]):.6f}"
            f"  ({time.time() - t0:.1f}s)")

    log("\n3) adjoint on tet10 (hybrid, incompressible elastin) with inclined supports: cylinder rotated by 30 deg")
    phi = np.radians(30.0)
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    m = meshlib.quarter_cylinder(n=(2, 15, 1), etype="tet10")
    m.X = m.X @ Rz.T
    sysm = fem.System(m, hcmm, pressure_faces=m.faces[(0, -1)], element="hybrid")
    Q = jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))
    ends = np.concatenate([3 * m.nodes[(2, -1)] + 2, 3 * m.nodes[(2, 1)] + 2])
    sym = np.concatenate([m.nodes[(1, -1)], m.nodes[(1, 1)]])
    normals = np.concatenate([np.tile(Rz @ [0, 1, 0], (len(m.nodes[(1, -1)]), 1)),
                              np.tile(Rz @ [1, 0, 0], (len(m.nodes[(1, 1)]), 1))])
    bc = lambda p: fem.BC(ends, np.zeros(len(ends)), p=p, slip=(sym, normals))
    idx = m.nodes[(0, -1)]
    e_r = m.X[idx, :2] / np.linalg.norm(m.X[idx, :2], axis=1, keepdims=True)
    a = float(np.linalg.norm(m.X[idx[0], :2]))
    measure = lambda u: (1 + (u.reshape(-1, 3)[idx, :2] * e_r).sum(1).mean() / a,)
    sim = lambda par: artery_simulate(m, sysm, Q, par, n_steps=20, bc=bc, measure=measure, pre_tol=1e-10, tol=1e-12,
                                      elastin="incompressible")[0][-1]
    par = artery_par()
    _, g = jax.value_and_grad(sim)(par)
    for k in ("C10", "k_plus", "p_gr"):
        h = 1e-5 * abs(float(par[k]))
        fd = (float(sim({**par, k: par[k] + h})) - float(sim({**par, k: par[k] - h}))) / (2 * h)
        err = abs(float(g[k]) - fd) / abs(fd)
        log(f"  {k:7s} adjoint {float(g[k]): .8e}   FD {fd: .8e}   rel err {err:.1e}  "
            f"{'PASS' if err < 1e-5 else 'FAIL'}")

    log.save(outputs("fe_gr/element_formulations", "element_formulations.txt")[0])
