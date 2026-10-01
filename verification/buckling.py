"""Buckling tools of fem.System against references (compressible neo-Hookean, plane strain via u_y = 0 for the arch):

    column   clamped-clamped, end shortening d:  stability() zero crossing  vs  Euler P_cr = 4 pi^2 E I / L^2
    arch     shallow clamped arch, central load:  arc_length() (force control, through the limit point)
             vs displacement control at the same deflections;  stability() sign change at the limit point;
             relaxation (System._ptc) from the limit point state at full load vs the arc length end state

Run: python -m verification.buckling
"""

import pathlib

import numpy as np

import fem
import materials
from fem import mesh as meshlib

C10, K = 0.305, 6.1
MU = 2 * C10
E = 9 * K * MU / (3 * K + MU)
OUT = pathlib.Path(__file__).parent / "results"


def column(a=1.0, L=40.0, n=(2, 2, 40)):
    """P_cr from the first zero of the smallest tangent eigenvalue along the end shortening d"""
    m = meshlib.box(n, (a, a, L), etype="tet10")
    sysm = fem.System(m)
    states = fem.broadcast_state(materials.NeoHookean(C10, K), m.n_elem, sysm.n_gp)
    base, top = m.nodes[(2, -1)], m.nodes[(2, 1)]
    fixed = np.concatenate([3 * base[:, None] + np.arange(3), 3 * top[:, None] + np.arange(3)]).ravel()
    zdofs = 3 * top + 2

    def at(d, u):
        vals = np.zeros(len(fixed))
        vals[3 * len(base) + 2::3] = -d
        bc = fem.BC(fixed, vals)
        u, _ = sysm.solve(u, states, bc)
        mu = sysm.stability(u, states, bc, k=1)[0][0]
        r = np.asarray(sysm.residual(u, states, 0.0, np.zeros(sysm.n_dof)))
        return mu, -r[zdofs].sum(), u

    P_euler = 4 * np.pi ** 2 * E * a ** 4 / 12 / L ** 2
    d_euler = P_euler * L / (E * a * a)
    u, prev = np.zeros(sysm.n_dof), None
    for d in np.linspace(0.5, 1.5, 11) * d_euler:
        mu, P, u = at(d, u)
        if prev is not None and prev[1] > 0 >= mu:
            d0, mu0 = prev[0], prev[1]
            break
        prev = (d, mu, u)
    lo, hi, ulo = d0, d, prev[2]
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        mu, P, um = at(mid, ulo)
        lo, hi, ulo = (mid, hi, um) if mu > 0 else (lo, mid, ulo)
    _, P_cr, _ = at(0.5 * (lo + hi), ulo)
    return P_cr, P_euler


def arch_mesh(S=40.0, H=2.0, t=1.0, b=1.0, n=(4, 60, 1)):
    """index axes (thickness, span, width)"""
    def mapping(s):
        x = (s[:, 1] - 0.5) * S
        return np.stack([x, b * s[:, 2], H * (1 - (2 * x / S) ** 2) + t * (s[:, 0] - 0.5)], axis=1)

    m = meshlib.structured(n, mapping)
    ends = np.union1d(m.nodes[(1, -1)], m.nodes[(1, 1)])
    ydofs = 3 * np.setdiff1d(np.arange(m.n_nodes), ends) + 1
    fixed = np.concatenate([(3 * ends[:, None] + np.arange(3)).ravel(), ydofs])
    load = np.flatnonzero((np.abs(m.X[:, 0]) < 1e-9) & (m.param[:, 0] == 1.0))
    return m, fixed, load


def arch(F=0.004):
    m, fixed, load = arch_mesh()
    sysm = fem.System(m)
    states = fem.broadcast_state(materials.NeoHookean(C10, K), m.n_elem, sysm.n_gp)
    f = np.zeros(sysm.n_dof)
    f[3 * load + 2] = -F / len(load)
    bc0 = fem.BC(fixed, np.zeros(len(fixed)))
    bc1 = fem.BC(fixed, np.zeros(len(fixed)), f_dead=f)
    path = sysm.arc_length(states, bc0, bc1, n_steps=40)
    lam = np.array([p[0] for p in path])
    w = np.array([-p[1][3 * load[0] + 2] for p in path])
    i_max = int(np.flatnonzero(np.diff(lam) < 0)[0])

    zd = 3 * load + 2
    fixed_d = np.concatenate([fixed, zd])
    u, err = np.zeros(sysm.n_dof), 0.0
    for lam_i, w_i in zip(lam[1:], w[1:]):
        bc = fem.BC(fixed_d, np.concatenate([np.zeros(len(fixed)), np.full(len(zd), -w_i)]))
        u, _ = sysm.solve(u, states, bc)
        R = -np.asarray(sysm.residual(u, states, 0.0, np.zeros(sysm.n_dof)))[zd].sum()
        err = max(err, abs(R - lam_i * F) / (lam[i_max] * F))

    mus = []
    for i in (i_max - 2, i_max + 2):
        mus.append(sysm.stability(path[i][1], states, replace_f(bc1, lam[i] * f), k=1)[0][0])

    try:
        sysm._newton(path[i_max][1], states, bc1, 1e-10, 30)
        newton = "converged"
    except RuntimeError:
        newton = "failed"
    x, it = sysm._ptc(path[i_max][1], states, bc1, 1e-10)
    e_end = np.abs(x[:sysm.n_dof] - path[-1][1]).max() / np.abs(path[-1][1]).max()
    return dict(lam_max=lam[i_max], w_max=w[i_max], lam_end=lam[-1], w_end=w[-1], n=len(path) - 1,
                err=err, mu_before=mus[0], mu_after=mus[1], newton=newton, ptc_it=it, e_end=e_end)


def replace_f(bc, f):
    return fem.BC(bc.fixed, bc.values, f_dead=f)


def main(tol=1e-6):
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    log(f"Buckling verification, neo-Hookean C10 {C10}, K {K} (E {E:.4f})")
    errs = []
    for n in ((2, 2, 40), (3, 3, 60)):
        P_cr, P_e = column(n=n)
        errs.append(abs(P_cr / P_e - 1))
        log(f"  column  tet10 {n[0]}x{n[1]}x{n[2]}, L/a 40:  P_cr {P_cr:.6f}  Euler {P_e:.6f}  rel. diff {errs[-1]:.2%}")
    ok_col = errs[-1] < 0.02 and errs[-1] < errs[0]
    log(f"    converges to Euler (rest: discretization, shear ~0.7 %)  {'PASS' if ok_col else 'FAIL'}")
    a = arch()
    ok_arc = a["err"] < tol
    ok_lim = a["mu_before"] > 0 > a["mu_after"]
    ok_ptc = a["e_end"] < 1e-6
    log(f"  arch    hex8 4x1x60, S 40, H 2, t 1:  {a['n']} arc length steps, limit load {a['lam_max']:.6f} "
        f"at w {a['w_max']:.4f}, end w {a['w_end']:.4f}")
    log(f"    arc length vs displacement control  max |F_arc - F_disp| / F_max {a['err']:.1e}  "
        f"{'PASS' if ok_arc else 'FAIL'}")
    log(f"    smallest eigenvalue before / after the limit point  {a['mu_before']:.3e} / {a['mu_after']:.3e}  "
        f"{'PASS' if ok_lim else 'FAIL'}")
    log(f"    full load from the limit state: Newton {a['newton']}, relaxation {a['ptc_it']} iterations, "
        f"end state vs arc length {a['e_end']:.1e}  {'PASS' if ok_ptc else 'FAIL'}")
    ok = ok_col and ok_arc and ok_lim and ok_ptc
    log("ALL PASS" if ok else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "buckling.txt").write_text("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    main()
