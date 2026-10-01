"""FE vs closed-form homogeneous deformations of neo-Hookean solids (Cauchy stress, stretch lam, shear g):

    uniaxial     incompressible, hybrid   F = diag(lam, lam^-1/2, lam^-1/2)   sigma_11 = 2 C10 (lam^2 - 1/lam)
    equibiaxial  incompressible, hybrid   F = diag(lam, lam, lam^-2)          sigma_11 = sigma_22 = 2 C10 (lam^2 - lam^-4)
    simple shear compressible             F = I + g e1(x)e2                   sigma = 2 C10 (B - I1/3 I)
    dilatation   compressible             F = a I                             sigma = K (a^3 - 1) I
    patch test   compressible             general F0                          sigma = sigma(F0) of the material

uniaxial/equibiaxial: symmetry planes + prescribed end faces, lateral faces free; shear/dilatation/patch test:
u = (F - I) X on the whole boundary.  Run: python -m verification.fem_analytic
"""

import pathlib

import jax.numpy as jnp
import numpy as np

import fem
import materials
from fem import mesh as meshlib

C10, K, L = 0.305, 6.1, 1.0
OUT = pathlib.Path(__file__).parent / "results"


def meshes():
    return {"hex8": meshlib.box((2, 2, 2), (L, L, L), perturb=0.2),
            "tet10": meshlib.box((2, 2, 2), (L, L, L), etype="tet10")}


def bc_faces(m, F, faces):
    """u_i = ((F - I) X)_i on the given (axis, side, component) faces"""
    vals = {}
    for axis, side, comp in faces:
        for n in m.nodes[(axis, side)]:
            vals[3 * n + comp] = ((F - np.eye(3)) @ m.X[n])[comp]
    dofs = np.array(sorted(vals))
    return fem.BC(dofs, np.array([vals[d] for d in dofs]))


def bc_boundary(m, F):
    faces = [(a, s, c) for a in range(3) for s in (-1, 1) for c in range(3)]
    return bc_faces(m, F, faces)


def check(m, material, F, sigma_exact, bc, element):
    sysm = fem.System(m, element=element)
    states = fem.broadcast_state(material, m.n_elem, sysm.n_gp)
    u, it = sysm.solve(np.zeros(sysm.n_dof), states, bc)
    q = sysm.last_p if sysm.n_p else np.zeros(m.n_elem)
    sig = np.asarray(sysm.stress(jnp.asarray(u), states)) - q[:, None, None, None] * np.eye(3)
    u_exact = ((F - np.eye(3)) @ m.X.T).T.ravel()
    eu = np.abs(u - u_exact).max() / L
    es = np.abs(sig - sigma_exact).max() / np.abs(sigma_exact).max()
    return eu, es, it


def cases():
    inc, comp = materials.NeoHookeanInc(C10), materials.NeoHookean(C10, K)
    sym = [(0, -1, 0), (1, -1, 1), (2, -1, 2)]
    for lam in (0.8, 1.3, 1.6):
        F = np.diag([lam, lam ** -0.5, lam ** -0.5])
        yield (f"uniaxial     lam={lam:<4}", inc, F, np.diag([2 * C10 * (lam ** 2 - 1 / lam), 0, 0]),
               lambda m, F=F: bc_faces(m, F, sym + [(0, 1, 0)]), ("hybrid",))
    for lam in (1.2, 1.4):
        F = np.diag([lam, lam, lam ** -2])
        s = 2 * C10 * (lam ** 2 - lam ** -4)
        yield (f"equibiaxial  lam={lam:<4}", inc, F, np.diag([s, s, 0]),
               lambda m, F=F: bc_faces(m, F, sym + [(0, 1, 0), (1, 1, 1)]), ("hybrid",))
    for g in (0.3, 1.0):
        F = np.eye(3) + g * np.outer([1, 0, 0], [0, 1, 0])
        B = F @ F.T
        yield (f"simple shear g={g:<4}", comp, F, 2 * C10 * (B - np.trace(B) / 3 * np.eye(3)),
               lambda m, F=F: bc_boundary(m, F), ("standard", "fbar"))
    for a in (0.95, 1.05):
        F = a * np.eye(3)
        yield (f"dilatation   a={a:<6}", comp, F, K * (a ** 3 - 1) * np.eye(3),
               lambda m, F=F: bc_boundary(m, F), ("standard", "fbar"))
    F = np.array([[1.08, 0.05, -0.02], [0.03, 0.95, 0.04], [-0.01, 0.02, 1.03]])
    yield ("patch test   general F0", comp, F, np.asarray(comp.sigma(jnp.asarray(F))) / np.linalg.det(F),
           lambda m: bc_boundary(m, F), ("standard", "fbar"))


def main(tol=1e-8):
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    log(f"FE vs closed-form homogeneous deformations, neo-Hookean C10 {C10}, K {K}; 2x2x2 box, hex8 interior perturbed")
    ok = True
    for name, material, F, sigma_exact, bc, elements in cases():
        for etype, m in meshes().items():
            for element in elements:
                eu, es, it = check(m, material, F, sigma_exact, bc(m), element)
                passed = eu < tol and es < tol
                ok &= passed
                log(f"  {name}  {etype:5s} {element:8s}  Newton {it:2d}  max|u - u_ex| {eu:.1e}  "
                    f"max rel sigma err {es:.1e}  {'PASS' if passed else 'FAIL'}")
    log("ALL PASS" if ok else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "fem_analytic.txt").write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    main()
