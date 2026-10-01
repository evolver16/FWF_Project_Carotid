"""Arterial G&R with the HCMM in the FE code (Maes & Famaey 2023 sec. 2.7, model E): quarter cylinder, gradients and
identification, pressure step (elastic vs turnover widening), pressure buckling of a G&R state."""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import fem
import hcmm
import materials
import plotting
import setups
from fem import mesh as meshlib
from fem.tensor3 import det3
from setups import ALPHA, FIBER, MATRIX_DE, G_fiber
from studies.common import Log, outputs, surface_area, write_csv
from verification.fem_tube import cylinder_bc, radial_u


# ---------- model E on a tube: parameters, states, prestress + G&R run ----------

def artery_par(**kw):
    """Model E on the artery: elastin (C10, K), fibers (k1, k2, k_plus, k_minus, g, turnover T in days),
    axial prestretch g_ax, pressures"""
    par = dict(C10=MATRIX_DE["C10"], K=MATRIX_DE["K"], k1=FIBER["k1"], k2=FIBER["k2"], k_plus=0.1, k_minus=0.0,
               T=setups.T_DAYS, g=FIBER["g"], g_ax=1.2, p_hom=0.010, p_gr=0.015)
    return {k: jnp.asarray(v, dtype=float) for k, v in {**par, **kw}.items()}


@partial(jax.jit, static_argnames=("mode", "elastin"))
def artery_states(G_e, Qloc, par, mode, ds=10.0, elastin="compressible"):
    """HCMM per Gauss point: elastin (no turnover) + fibers along e_theta, e_z, +-alpha; growth along e_r.
    elastin "incompressible" needs the hybrid element."""

    def make(Ge, Q):
        e_r, e_t, e_z = Q[:, 0], Q[:, 1], Q[:, 2]
        dirs = [e_t, e_z, jnp.cos(ALPHA) * e_t + jnp.sin(ALPHA) * e_z, jnp.cos(ALPHA) * e_t - jnp.sin(ALPHA) * e_z]
        mat = (materials.NeoHookean(par["C10"], par["K"]) if elastin == "compressible"
               else materials.NeoHookeanInc(par["C10"]))
        cs = [hcmm.constituent(mat, MATRIX_DE["rho_0"], G=Ge, sigma_pre_mode=mode)]
        cs += [hcmm.constituent(materials.Fung(par["k1"], par["k2"], M), FIBER["rho_0"],
                                *hcmm.maes(par["T"], par["k_plus"], par["k_minus"]), G=G_fiber(par["g"], M),
                                sigma_pre_mode=mode) for M in dirs]
        return hcmm.mixture(cs, ds=ds, growth=hcmm.Anisotropic(e_r))

    st = jax.vmap(jax.checkpoint(make))(G_e.reshape(-1, 3, 3), Qloc.reshape(-1, 3, 3))
    return jax.tree.map(lambda x: x.reshape(G_e.shape[:2] + x.shape[1:]), st)


def artery_mesh(n=(8, 60, 4), element="standard"):
    """(mesh, System, local bases (e_r, e_theta, e_z) at the Gauss points)"""
    m = meshlib.quarter_cylinder(n=n)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces[(0, -1)], element=element)
    return m, sysm, jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))


def wall_area(m, u):
    """4 x area of the deformed end face z = 0"""
    return 4 * surface_area(m, u, m.faces[(2, -1)])


def artery_simulate(m, sysm, Qloc, par, mode="deposition", n_steps=100, bc=None, measure=None,
                    pre_tol=1e-6, pre_iter=60, anderson=5, tol=1e-10, log=None, final=False,
                    elastin="compressible", checkpoint=None, record=None, bc_gr=None, insult=None, ds=10.0):
    """Prestress G_elas <- F G_elas at p_hom until u = 0 (Maes UMAT_DEP; Anderson-accelerated, 0 = plain),
    G&R at p_gr, steps of ds days -> measure(u) per step (default lambda_theta(inner), wall area),
    reverse-differentiable in par; final: also (u, states) at the end; record(k, u, states): after prestress (k = 0) and step k (forward only);
    bc_gr: G&R BCs per step instead of bc(p_gr); insult(states) -> states: applied once after the prestress"""
    if bc is None:
        bc = lambda p: cylinder_bc(m, p)
    if measure is None:
        a = float(np.linalg.norm(m.X[m.nodes[(0, -1)][0], :2]))
        measure = lambda u: (1 + radial_u(m, u, -1) / a, wall_area(m, u))
    g_ax = par["g_ax"]
    G_e = Qloc @ jnp.diag(jnp.stack([g_ax ** -0.5, g_ax ** -0.5, g_ax])) @ jnp.swapaxes(Qloc, -1, -2)
    zero = jnp.zeros(sysm.n_dof)
    xs, fs = [], []
    for it in range(1, pre_iter + 1):
        states = artery_states(G_e, Qloc, par, mode, ds=ds, elastin=elastin)
        u = sysm.equilibrium(states, bc(par["p_hom"]), zero, tol)
        du = float(np.abs(sysm.last_u).max())
        if log and (it % 5 == 0 or du < pre_tol):
            log(f"  prestress iteration {it:2d}  max|u| {du:.2e} mm  (Newton {sysm.last_iterations})")
        if du < pre_tol:
            break
        xs, fs = (xs + [G_e.ravel()])[-anderson - 1:], (fs + [(sysm.gauss_F(u) @ G_e - G_e).ravel()])[-anderson - 1:]
        G_e = fem.anderson(xs, fs).reshape(G_e.shape)

    if insult:
        states = insult(states)
    on_step = lambda u, _: measure(u)
    if record:
        record(0, u, states)
        k = iter(range(1, n_steps + 1))
        on_step = lambda u, st: (record(next(k), u, st), measure(u))[1]
    u, states, hist = sysm.run(states, bc_gr or [bc(par["p_gr"])] * n_steps, u0=zero, tol=tol,
                               on_step=on_step, checkpoint=checkpoint)
    series = tuple(jnp.stack(v) for v in zip(*hist))
    return (series, u, states) if final else series


# ---------- G&R of the quarter cylinder (Maes & Famaey 2023 Fig. 4) ----------

def artery_gr(mode="deposition", n=(8, 60, 4), n_steps=100):
    log = Log()
    m, sysm, Qloc = artery_mesh(n)
    par = artery_par()
    log(f"mesh {n}: {m.n_elem} hex8, {sysm.n_dof} dofs, HCMM model E, sigma_pre mode '{mode}'")
    log(f"prestress at {float(par['p_hom']) * 1e3:.0f} kPa, then G&R at {float(par['p_gr']) * 1e3:.0f} kPa, "
        f"{n_steps} steps of 10 days   (reference area {float(wall_area(m, np.zeros(sysm.n_dof))):.3f} mm^2)")
    t0 = time.time()
    lam, area = map(np.asarray, artery_simulate(m, sysm, Qloc, par, mode, n_steps, log=log))
    for k in (0, 9, 24, 49, 74, 99):
        if k < n_steps:
            log(f"  day {10 * (k + 1):4d}  lambda_theta(inner) {lam[k]:.4f}  area {area[k]:.3f} mm^2")
    log(f"  done in {time.time() - t0:.1f}s")

    tag = "dep" if mode == "deposition" else "init"
    txt, csv_path, fig = outputs("artery/artery_gr", f"artery_{tag}.txt", f"artery_{tag}.csv", f"artery_{tag}.png")
    write_csv([dict(day=10 * (k + 1), lambda_theta=float(lam[k]), area=float(area[k])) for k in range(n_steps)], csv_path)
    t = np.arange(1, n_steps + 1) * 10.0
    plotting.panels([dict(t=t, series={f"HCMM {tag}": lam}, title="Inner circumferential stretch", ylabel="lambda_theta (-)"),
                     dict(t=t, series={f"HCMM {tag}": area}, title="Cross-sectional area", ylabel="area (mm^2)")], fig)
    log.save(txt)


# ---------- gradients through prestress + G&R, parameter identification ----------

def artery_gradients(n_check=(2, 12, 1), steps_check=20, n_full=(8, 60, 4), steps_full=100):
    """Adjoint gradients of lambda_theta(end) vs central differences, cost on the full mesh, identification."""
    import scipy.optimize as so
    log = Log()
    names = ["C10", "K", "k1", "k2", "k_plus", "g", "g_ax", "p_gr"]
    m, sysm, Qloc = artery_mesh(n_check)
    sim = lambda par: artery_simulate(m, sysm, Qloc, par, n_steps=steps_check, pre_tol=1e-10, tol=1e-12)
    par = artery_par()

    log(f"1) d lambda_theta(day {10 * steps_check}) / d par, mesh {n_check}, prestress to 1e-10 mm: adjoint vs central FD")
    t0 = time.time()
    val, g = jax.value_and_grad(lambda p: sim(p)[0][-1])(par)
    log(f"  lambda_theta {float(val):.8f}   value + gradient in {time.time() - t0:.1f}s")
    for k in names:
        h = 1e-5 * abs(float(par[k]))
        fd = (sim({**par, k: par[k] + h})[0][-1] - sim({**par, k: par[k] - h})[0][-1]) / (2 * h)
        err = abs(float(g[k]) - float(fd)) / max(abs(float(fd)), 1e-12)
        log(f"  {k:7s} adjoint {float(g[k]): .8e}   FD {float(fd): .8e}   rel err {err:.1e}  "
            f"{'PASS' if err < 1e-5 else 'FAIL'}")

    log(f"\n2) identification of (k1, k_plus) from lambda_theta(t), mesh {n_check}, L-BFGS in log space")
    true = artery_par()
    data = sim(true)[0]
    x_ref = {k: true[k] for k in ("k1", "k_plus")}

    def loss(x):
        p = {**true, **{k: x_ref[k] * jnp.exp(x[i]) for i, k in enumerate(x_ref)}}
        return 1e6 * jnp.sum((sim(p)[0] - data) ** 2)

    vg = jax.value_and_grad(loss)
    trace = []

    def fun(x):
        v, dv = vg(jnp.asarray(x))
        trace.append(float(v))
        return float(v), np.asarray(dv)

    x0 = np.log([0.7, 0.5])
    t0 = time.time()
    res = so.minimize(fun, x0, jac=True, method="L-BFGS-B", options=dict(ftol=1e-16, gtol=1e-10, maxiter=40))
    est = {k: float(x_ref[k]) * np.exp(res.x[i]) for i, k in enumerate(x_ref)}
    log(f"  start k1 {float(true['k1']) * 0.7:.4f}  k_plus {float(true['k_plus']) * 0.5:.4f}   loss {trace[0]:.3e}")
    log(f"  found k1 {est['k1']:.6f}  k_plus {est['k_plus']:.6f}   loss {res.fun:.3e}   "
        f"(true {float(true['k1']):.6f}, {float(true['k_plus']):.6f})   {len(trace)} evaluations, {time.time() - t0:.1f}s")

    log(f"\n3) cost on mesh {n_full}, {steps_full} steps")
    m, sysm, Qloc = artery_mesh(n_full)
    run = lambda par: artery_simulate(m, sysm, Qloc, par, n_steps=steps_full)[0][-1]
    t0 = time.time()
    val = run(par)
    t_fwd = time.time() - t0
    t0 = time.time()
    val, g = jax.value_and_grad(run)(par)
    t_grad = time.time() - t0
    log(f"  forward {t_fwd:.1f}s   value + gradient w.r.t. {len(par)} parameters {t_grad:.1f}s   "
        f"(central FD would need {2 * len(par)} forward runs, ~{2 * len(par) * t_fwd:.0f}s)")
    log(f"  lambda_theta(day {10 * steps_full}) {float(val):.6f}")
    for k in par:
        log(f"  d/d{k:7s} {float(g[k]): .6e}   elasticity (p/lam) d lam/dp {float(g[k] * par[k] / val): .4f}")

    log.save(outputs("artery/artery_gradients", "gradients.txt")[0])


# ---------- pressure step: elastic vs turnover widening ----------

def pressure_step(days=2000.0, n_steps=200, n=(4, 30, 1), name="pressure_step", **par_kw):
    """Pressure step p_hom -> p_gr held constant, uniform quarter cylinder: widening = elastic part (equilibrium at
    p_gr before any turnover) + turnover part (the rest).  par_kw: any artery_par key (p_gr, T, k_plus, k_minus, ...)"""
    log = Log()
    txt, pvd, csv_path, fig = outputs("artery/pressure_step", *(f"{name}{s}" for s in (".txt", ".pvd", ".csv", ".png")))
    par = artery_par(**par_kw)
    ds = days / n_steps
    m = meshlib.quarter_cylinder(n=n)
    sysm = fem.System(m, hcmm, pressure_faces=m.faces[(0, -1)])
    Qj = jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))
    bc = lambda p: cylinder_bc(m, p)
    radii = lambda u: (5.0 + float(radial_u(m, u, -1)), 6.3 + float(radial_u(m, u, 1)))
    w = np.asarray(sysm.wdet)
    hist, rho0, elastic = [], [], {}
    series = meshlib.VTKSeries(pvd)

    def record(k, u, states):
        """after the commit of step k; k = 0 also: elastic response at p_gr with the prestressed state"""
        st = sysm.commit(u, states) if k else states
        rho = [np.asarray(c.rho) for c in st.constituents]
        if not k:
            rho0.extend(rho)
            elastic["r_in"], elastic["r_out"] = radii(sysm.equilibrium(states, bc(par["p_gr"]), u))
        col, col0 = sum(rho[1:]), sum(rho0[1:])
        s_rel = np.asarray(st.constituents[1].sigma_f / st.constituents[1].sigma_f_pre)
        r_in, r_out = radii(u)
        hist.append(dict(day=ds * k, r_in=r_in, h=r_out - r_in, collagen=float(np.sum(w * col) / np.sum(w * col0)),
                         sigma_circ_rel=float(np.sum(w * s_rel) / w.sum())))
        series.write(ds * k, m.X, m.conn, *sysm.fields(u, states, dict(
            J_g=det3(st.F_g), rho_rel_collagen=col / col0, sigma_circ_rel=s_rel)))

    log(f"quarter cylinder r_i 5, t 1.3 mm: {m.n_elem} hex8, {sysm.n_dof} dofs, HCMM model E")
    log("parameters: " + "  ".join(f"{k} {float(v):.4g}" for k, v in par.items()))
    log(f"pressure {float(par['p_hom']) * 1e3:.1f} -> {float(par['p_gr']) * 1e3:.1f} kPa, held for {days:.0f} days, "
        f"{n_steps} steps of {ds:.3g} days (ds/T {ds / float(par['T']):.3f})")
    t0 = time.time()
    artery_simulate(m, sysm, Qj, par, n_steps=n_steps, bc=bc, log=log, record=record,
                    measure=lambda u: (jnp.zeros(()),), ds=ds)
    log(f"  done in {time.time() - t0:.1f}s, ParaView: {pvd.name} ({len(series.entries)} time steps)")

    r0, h0 = hist[0]["r_in"], hist[0]["h"]
    el = elastic["r_in"] / r0 - 1
    log(f"\nelastic widening at {float(par['p_gr']) * 1e3:.1f} kPa (no turnover): inner radius {100 * el:+.2f} %, "
        f"wall thickness {100 * ((elastic['r_out'] - elastic['r_in']) / h0 - 1):+.2f} %")
    log("\n    day   inner radius: total   elastic   turnover   wall thickness   collagen mass   circ. collagen "
        "stress / set point")
    for k in sorted({int(round(f * n_steps)) for f in (0, 0.005, 0.05, 0.1, 0.25, 0.5, 0.75, 1)}):
        r = hist[k]
        tot = r["r_in"] / r0 - 1
        log(f"  {r['day']:5.0f}   {100 * tot:+17.2f} % {100 * el * (k > 0):+8.2f} % {100 * (tot - el) * (k > 0):+8.2f} %"
            f"   {100 * (r['h'] / h0 - 1):+12.2f} %   {100 * (r['collagen'] - 1):+11.2f} %      {r['sigma_circ_rel']:.4f}")

    write_csv([r | dict(r_in_elastic=elastic["r_in"] if r["day"] else r0) for r in hist], csv_path)
    t = np.array([r["day"] for r in hist])
    r_in = np.array([r["r_in"] for r in hist]) / r0 - 1
    plotting.panels([dict(t=t[1:], series={"total": 100 * r_in[1:], "elastic": np.full(len(t) - 1, 100 * el),
                                           "turnover": 100 * (r_in[1:] - el)},
                          title="Inner radius change", ylabel="%"),
                     dict(t=t, series={"wall thickness": 100 * (np.array([r["h"] for r in hist]) / h0 - 1),
                                       "collagen mass": 100 * (np.array([r["collagen"] for r in hist]) - 1)},
                          title="Wall", ylabel="%"),
                     dict(t=t, series={"circ. collagen": np.array([r["sigma_circ_rel"] for r in hist])},
                          title="Collagen stress / set point", ylabel="(-)")], fig)
    log.save(txt)


# ---------- pressure buckling of a G&R state, G&R frozen ----------

def pressure_buckling(p_max=0.08, eps=0.05, L=60.0, n=(2, 16, 40), n_gr=5, g_ax=1.2, name="pressure_buckling"):
    """Pressure buckling of a G&R state with G&R frozen: half tube, ends fixed, prestress at p_hom + n_gr steps at
    p_gr; stability() along the perfect tube -> critical pressure and mode; geometry + eps * mode (max |mode| = 1),
    prestress and G&R redone, arc_length p_gr -> p_max -> ParaView series over the pressure (kPa)"""
    log = Log()
    txt, pvd, csv_path = outputs("artery/pressure_buckling", *(f"{name}{s}" for s in (".txt", ".pvd", ".csv")))
    par = artery_par(g_ax=g_ax)
    p_gr = float(par["p_gr"])

    def setup(mode=None):
        m = meshlib.half_tube(L=L, n=n)
        if mode is not None:
            m.X = m.X + eps * mode
        sysm = fem.System(m, hcmm, pressure_faces=m.faces["inner"])
        Q = jnp.asarray(meshlib.cylinder_basis(fem.gauss_points(m.X, m.conn)))
        ends = np.union1d(m.nodes["inlet"], m.nodes["outlet"])
        fixed = np.unique(np.concatenate([(3 * ends[:, None] + np.arange(3)).ravel(), 3 * m.nodes["sym"] + 1]))
        bc = lambda p: fem.BC(fixed, np.zeros(len(fixed)), p=float(p))
        _, u, states = artery_simulate(m, sysm, Q, par, n_steps=n_gr, bc=bc, measure=lambda u: (jnp.zeros(()),),
                                       final=True)
        return m, sysm, bc, u, states

    pressure = lambda lam: p_gr + lam * (p_max - p_gr)
    log(f"half tube r_i 5, t 1.3, L {L} mm, ends fixed, n {n}; HCMM model E, g_ax {g_ax}; prestress at "
        f"{float(par['p_hom']) * 1e3:.0f} kPa, {n_gr} G&R steps at {p_gr * 1e3:.0f} kPa, then G&R frozen")
    t0 = time.time()
    m, sysm, bc, u, states = setup()
    mid = np.flatnonzero(np.abs(m.X[:, 2] - L / 2) < 1e-9)
    defl = lambda u: float(np.asarray(u).reshape(-1, 3)[mid, 0].mean())
    path = sysm.arc_length(states, bc(p_gr), bc(p_max), u0=u)
    mus = [sysm.stability(v, states, bc(pressure(lam)), k=1)[0][0] for lam, v in path]
    i = next((k for k in range(1, len(mus)) if mus[k] < 0 <= mus[k - 1]), None)
    if i is None:
        log(f"perfect tube: no instability up to {p_max * 1e3:.0f} kPa")
        log.save(txt)
        return
    lam_c = path[i - 1][0] + mus[i - 1] / (mus[i - 1] - mus[i]) * (path[i][0] - path[i - 1][0])
    mode = sysm.stability(path[i][1], states, bc(pressure(path[i][0])), k=1)[1][0].reshape(-1, 3)
    log(f"perfect tube: smallest eigenvalue crosses 0 at p_cr ~ {pressure(lam_c) * 1e3:.1f} kPa "
        f"(mode: centerline x at mid-length {mode[mid, 0].mean():.2f}, max |u_z| {np.abs(mode[:, 2]).max():.2f})")

    m, sysm, bc, u, states = setup(mode)
    mid = np.flatnonzero(np.abs(m.X[:, 2] - L / 2) < 1e-6)
    series = meshlib.VTKSeries(pvd)
    rows = []

    def write(lam, v):
        p = pressure(lam)
        mu = sysm.stability(v, states, bc(p), k=1)[0][0]
        rows.append(dict(p_kPa=p * 1e3, deflection=defl(v), mu_min=mu))
        pd, cd = sysm.fields(v, states)
        series.write(p * 1e3, m.X, m.conn, dict(pd, mode=mode), cd)

    path = sysm.arc_length(states, bc(p_gr), bc(p_max), u0=u)
    for lam, v in path:
        write(lam, v)
    log(f"imperfection {eps} mm x mode: {len(path) - 1} arc length steps to {p_max * 1e3:.0f} kPa, "
        f"{time.time() - t0:.0f}s, ParaView: {pvd.name} (time = pressure in kPa)")
    log("\n   p (kPa)   centerline deflection (mm)   smallest eigenvalue")
    for r in rows:
        log(f"  {r['p_kPa']:8.1f}   {r['deflection']:12.3f}              {r['mu_min']: .2e}")
    write_csv(rows, csv_path)
    log.save(txt)
