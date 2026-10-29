"""Reduced artery model: network of geometrically exact C1 rods (cubic Hermite centerline in 3D), each section a thin
ring of material points of a fem-style G&R model (sigma_solver, J_target, commit), branches joined at rigid junctions,
optionally with a junction wall (membrane material points on the junction surface, moved by the rod sections)."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

import fem

XG, WG = np.polynomial.legendre.leggauss(3)
XG, WG = (XG + 1) / 2, WG / 2
H0 = np.stack([1 - 3 * XG ** 2 + 2 * XG ** 3, XG - 2 * XG ** 2 + XG ** 3, 3 * XG ** 2 - 2 * XG ** 3, -XG ** 2 + XG ** 3], -1)
H1 = np.stack([-6 * XG + 6 * XG ** 2, 1 - 4 * XG + 3 * XG ** 2, 6 * XG - 6 * XG ** 2, -2 * XG + 3 * XG ** 2], -1)
H2 = np.stack([-6 + 12 * XG, -4 + 6 * XG, 6 - 12 * XG, -2 + 6 * XG], -1)


def rotation(phi):
    """R = exp([phi]x) (Rodrigues), smooth at phi = 0"""
    th2 = phi @ phi
    small = th2 < 1e-8
    th = jnp.sqrt(jnp.where(small, 1.0, th2))
    a = jnp.where(small, 1 - th2 / 6, jnp.sin(th) / th)
    b = jnp.where(small, 0.5 - th2 / 24, (1 - jnp.cos(th)) / th ** 2)
    K = jnp.cross(phi, jnp.eye(3)).T
    return jnp.eye(3) + a * K + b * K @ K


def section_map(z, L0, T0, X0, A_end, X):
    """motion of a reference point X by the rod element z (r_i, t_i, r_j, t_j, a_i, a_j) through its section at the
    projection xi of X (radial offset scaled by a / A and turned by the minimal rotation T_0 -> t; beyond the element
    ends the end section moves rigidly along t)"""
    s = (X - X0) @ T0
    xi = s / L0
    x_ = jnp.clip(xi, 0.0, 1.0)
    h0 = jnp.array([1 - 3 * x_ ** 2 + 2 * x_ ** 3, x_ - 2 * x_ ** 2 + x_ ** 3, 3 * x_ ** 2 - 2 * x_ ** 3, -x_ ** 2 + x_ ** 3])
    h1 = jnp.array([-6 * x_ + 6 * x_ ** 2, 1 - 4 * x_ + 3 * x_ ** 2, 6 * x_ - 6 * x_ ** 2, -2 * x_ + 3 * x_ ** 2])
    E = jnp.stack([z[0:3], z[3:6], z[6:9], z[9:12]]) * jnp.array([1.0, L0, 1.0, L0])[:, None]
    x, d1 = h0 @ E, h1 @ E / L0
    t = d1 / jnp.linalg.norm(d1)
    rv = X - X0 - s * T0
    c = jnp.cross(T0, t)
    rv = rv + jnp.cross(c, rv) + jnp.cross(c, jnp.cross(c, rv)) / (1 + T0 @ t)
    a, A = z[12] * (1 - x_) + z[13] * x_, A_end[0] * (1 - x_) + A_end[1] * x_
    return x + a / A * rv + (xi - x_) * L0 * d1


class Rod:
    """Straight elements between centerline nodes X (n, 3), conn (n_el, 2), ordered and equally directed along each
    branch (branch id per element); reference radius A at the element ends (n_el, 2), linear along the element;
    3 Gauss sections per element with wall thickness H (n_el, 3), n_ring material points around each (angle theta
    from D1 = e_y x T_0, D2 = T_0 x D1).
    DOFs: node position r and tangent r' = dr/dS; at rigid nodes (junctions) the element-end tangents turn together,
    t = lambda R(phi) T_0 (phi, lambda per end free), clamped nodes: r fixed, t = lambda T_0; radius a per node
    (per element end at rigid nodes), linear along the element. Ring point kinematics (local basis e_r, e_theta, e_z;
    ring turned by the minimal rotation T_0 -> t, no twist):
        lambda_theta = a / A,  lambda_z = sqrt((|r'| - a e(theta) . t')^2 + a'^2) / sqrt(1 + A'^2),
        lambda_r = J_g / (lambda_theta lambda_z)
    (thin wall, sigma_rr = 0; the slope a' = da/dS couples neighbouring rings: a bulge stretches the wall axially);
    equilibrium sum J_g dV_0 (s_tt d ln lambda_theta + s_zz d ln lambda_z) = p dV, V = int pi a^2 ds (lumen volume:
    the pressure load p pi a^2 of a curved tube).
    offsets (n_el, 2, 3): element end = node + offset, turned with the node at rigid nodes (a junction body, e.g.
    branch axes starting beside the junction node); active (n_el,): elements with wall and lumen (False: kinematics
    only, 1e-3 of the wall against singular radii; the junction wall takes over). States per material point, shape
    (n_points, 1): ring points (element, section, ring), then junction points (attach_junction)."""

    def __init__(self, X, conn, branch, A, H, model, rigid=(), clamped=(), n_ring=16, offsets=None, active=None):
        self.X, self.conn, self.branch, self.model = np.asarray(X, float), np.asarray(conn), np.asarray(branch), model
        n, ne, ng = len(self.X), len(self.conn), len(XG)
        self.n_el, self.n_ring = ne, n_ring
        off = np.zeros((ne, 2, 3)) if offsets is None else np.asarray(offsets, float)
        self.Xe = self.X[self.conn] + off
        d = self.Xe[:, 1] - self.Xe[:, 0]
        L0 = np.linalg.norm(d, axis=1)
        T0 = d / L0[:, None]
        D1 = np.cross([0.0, 1.0, 0.0], T0)
        D1 /= np.linalg.norm(D1, axis=1, keepdims=True)
        D2 = np.cross(T0, D1)
        th = (np.arange(n_ring) + 0.5) * 2 * np.pi / n_ring
        self.th, self.cos_th, self.sin_th = th, jnp.asarray(np.cos(th)), jnp.asarray(np.sin(th))
        A_end = np.asarray(A, float)
        self.A, self.H = A_end[:, :1] * (1 - XG) + A_end[:, 1:] * XG, np.asarray(H, float)
        self.active = np.ones(ne, bool) if active is None else np.asarray(active, bool)
        self.geo = tuple(jnp.asarray(v) for v in (L0, T0, D1, D2, A_end, self.active.astype(float)))
        self.dV0 = jnp.asarray(np.repeat((WG * L0[:, None] * self.A * self.H * 2 * np.pi / n_ring
                                          * np.where(self.active, 1.0, 1e-3)[:, None])[..., None], n_ring, -1))
        ring = np.cos(th)[None, :, None] * D1[:, None] + np.sin(th)[None, :, None] * D2[:, None]
        xs = self.Xe[:, 0][:, None] + XG[None, :, None] * d[:, None]
        self.X_points = (xs[:, :, None] + self.A[..., None, None] * ring[:, None]).reshape(-1, 3)
        self.n_rp = len(self.X_points)
        self.dV_points = np.asarray(self.dV0).ravel()
        self.T0_points = np.repeat(np.repeat(T0[:, None], ng, 1)[:, :, None], n_ring, 2).reshape(-1, 3)
        rigid_all = np.zeros(n, bool)
        rigid_all[list(rigid) + list(clamped)] = True
        tau0 = np.zeros((n, 3))
        tau0[self.conn[:, 1]], tau0[self.conn[:, 0]] = T0, T0
        slots = {}
        self.slot = np.array([[slots.setdefault((nd, e) if rigid_all[nd] else nd, len(slots))
                               for nd in self.conn[e]] for e in range(ne)])
        a0 = np.zeros(len(slots))
        a0[self.slot] = A_end
        base = np.concatenate([self.X.ravel(), tau0.ravel(), np.zeros(3 * n), np.ones(2 * ne), a0])
        idx = np.arange(9 * n).reshape(3, n, 3)
        fix = np.zeros(len(base), bool)
        fix[idx[0][list(clamped)].ravel()] = True
        fix[idx[1][rigid_all].ravel()] = True
        fix[idx[2][~np.isin(np.arange(n), list(rigid))].ravel()] = True
        fix[9 * n + np.flatnonzero(~rigid_all[self.conn].ravel())] = True
        self.free = np.flatnonzero(~fix)
        self.base, self.q0, self.rigid = jnp.asarray(base), jnp.asarray(base[self.free]), jnp.asarray(rigid_all)
        self.off = jnp.asarray(off)
        self._nl = {}
        nz = 14
        self._rows = (np.arange(ne)[:, None, None] * nz + np.arange(nz)[:, None] + 0 * np.arange(nz)).ravel()
        self._cols = (np.arange(ne)[:, None, None] * nz + 0 * np.arange(nz)[:, None] + np.arange(nz)).ravel()
        self._flat = lambda st, sl: jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:])[sl], st)
        self._per_el = lambda st: jax.tree.map(lambda x: x.reshape(ne, ng, n_ring, *x.shape[1:]),
                                               self._flat(st, slice(0, self.n_rp)))
        self._expand = jax.jit(self.expand)
        self.nl = np.flatnonzero((self.free >= 6 * n) & (self.free < 9 * n + 2 * ne))
        part = lambda f: (lambda q, *a: f(lambda y: self.expand(q.at[self.nl].set(y)), q[self.nl], *a))
        self._J_nl = jax.jit(part(lambda g, y: jax.jacfwd(g)(y)))
        self._G_nl = jax.jit(part(lambda g, y, rz: jax.hessian(lambda y: jnp.sum(rz * g(y)))(y)))
        J0 = np.array(jax.jacfwd(self.expand)(self.q0)).reshape(ne * nz, -1)
        J0[:, self.nl] = 0.0
        self._J0 = sp.csr_matrix(J0)
        self._res_e = jax.jit(jax.vmap(self._residual_e, (0, 0, 0, 0, None)))
        self._K_e = jax.jit(jax.vmap(jax.jacfwd(self._residual_e), (0, 0, 0, 0, None)))
        self.ties, self.tie_L0, self.tie_Lr, self._tie_q = np.zeros((0, 2), int), np.zeros(0), np.zeros(0), None
        bar = lambda y, L0, Lr, EA: 0.5 * EA * (jnp.linalg.norm(y[3:] - y[:3]) - Lr) ** 2 / L0
        self._tie_g = jax.jit(jax.vmap(jax.grad(bar)))
        self._tie_K = jax.jit(jax.vmap(jax.hessian(bar)))
        self.junction = None
        self._compile()

    def _compile(self):
        self._rz = jax.jit(self._rz_impl)
        self._residual = jax.jit(lambda q, st, p: jax.vjp(self.expand, q)[1](self._rz(self.expand(q), st, p))[0])
        self.F = jax.jit(self._F)
        self.stresses = jax.jit(lambda q, st: self._wall(st, self._F(q, st), 2))
        self.commit = jax.jit(lambda q, st: (lambda F: jax.vmap(jax.vmap(self.model.commit))(
            st, F, jax.vmap(jax.vmap(self.model.sigma_solver))(st, F)[1]))(self._F(q, st)))
        if self.junction is not None:
            J = self.junction
            self._KJ = jax.jit(lambda z, st, p: jax.vmap(jax.jacfwd(self._jres), (0, 0, 0, None))(
                z[J["elems"]], J["pd"], self._flat(st, slice(self.n_rp, None)), p))
            self._Kd = jax.jit(lambda z, p: -p * jax.vmap(jax.hessian(self._disc))(z[J["disc_el"]], J["disc_k"],
                                                                                   J["disc_s"]))

    def attach_junction(self, X, N, Ez, dA, H, elems, w, discs):
        """junction wall: material points X (n, 3) on the reference junction surface, outward normals N, axial
        directions Ez (local basis e_r = N, e_theta = Ez x N, e_z = Ez), areas dA, thickness H; elems (n, k): the
        element of each tube that moves the point, w (n, k) blending weights (rows sum to 1),
        Phi(X) = sum_k w_k section_map(z_k, X); membrane: g_a = grad Phi E_a, F in the local basis (upper triangular
        in-plane block, lambda_r = J_g / det), sigma_rr = 0; discs (element, end, sign) close the junction lumen at
        the rod sections where it ends (sign: outward along t), V = 1/3 (sum x . n dA + discs)"""
        L0, T0, _, _, A_end, _ = self.geo
        el = jnp.asarray(elems)
        X, N, Ez = (np.asarray(v, float) for v in (X, N, Ez))
        pd = (L0[el], T0[el], jnp.asarray(self.Xe[:, 0])[el], A_end[el], jnp.asarray(X), jnp.asarray(np.cross(Ez, N)),
              jnp.asarray(Ez), jnp.asarray(w, float), jnp.asarray(dA * H, float), jnp.asarray(dA, float))
        disc = np.asarray(discs)
        self.junction = dict(elems=el, pd=pd, n=len(X), X=X, disc_el=jnp.asarray(disc[:, 0].astype(int)),
                             disc_k=jnp.asarray(disc[:, 1].astype(int)), disc_s=jnp.asarray(disc[:, 2], float))
        self.X_points = np.concatenate([self.X_points[:self.n_rp], X])
        self.dV_points = np.concatenate([self.dV_points[:self.n_rp], np.asarray(dA * H, float)])
        self._nl = {}
        self._compile()

    def expand(self, q):
        """per element (r_i, t_i, r_j, t_j, a_i, a_j)"""
        n, ne = len(self.X), self.n_el
        v = self.base.at[self.free].set(q)
        r, tau, phi = v[:3 * n].reshape(n, 3), v[3 * n:6 * n].reshape(n, 3), v[6 * n:9 * n].reshape(n, 3)
        lam, a = v[9 * n:9 * n + 2 * ne].reshape(ne, 2), v[9 * n + 2 * ne:][self.slot]
        Rn = jax.vmap(rotation)(phi)
        out = []
        for k in (0, 1):
            nd = self.conn[:, k]
            rig = self.rigid[nd][:, None]
            t = jnp.where(rig, lam[:, k, None] * jnp.einsum("eij,ej->ei", Rn[nd], self.geo[1]), tau[nd])
            out += [r[nd] + jnp.where(rig, jnp.einsum("eij,ej->ei", Rn[nd], self.off[:, k]), 0.0), t]
        return jnp.concatenate(out + [a], 1)

    def tie(self, ties, EA):
        """elastic bars between free nodes (e.g. a bifurcation wall idealized as a triangle between the branches):
        N = EA (l - L_r) / L_0, L_0 reference length, rest length L_r set by the prestress (N carries the homeostatic
        load at the reference)"""
        self.ties = np.asarray(ties)
        d = self.X[self.ties[:, 1]] - self.X[self.ties[:, 0]]
        self.tie_L0 = np.linalg.norm(d, axis=1)
        self.tie_Lr = self.tie_L0.copy()
        self.tie_EA = np.broadcast_to(np.asarray(EA, float), len(self.ties)).copy()
        where = {f: i for i, f in enumerate(self.free)}
        self._tie_q = np.array([[where[3 * n + c] for n in t for c in range(3)] for t in self.ties])

    def tie_forces(self, q):
        y = np.asarray(q)[self._tie_q]
        return self.tie_EA * (np.linalg.norm(y[:, 3:] - y[:, :3], axis=1) - self.tie_Lr) / self.tie_L0

    def residual(self, q, st, p):
        r = self._residual(q, st, p)
        if len(self.ties):
            g = self._tie_g(jnp.asarray(q)[self._tie_q], self.tie_L0, self.tie_Lr, self.tie_EA)
            r = r.at[self._tie_q.ravel()].add(g.ravel())
        return r

    def positions(self, q):
        return np.asarray(self.base.at[self.free].set(q)[:3 * len(self.X)]).reshape(-1, 3)

    def _element(self, z, L0, T0, D1, D2, A_end, vw):
        """(strains ln lambda_theta, ln lambda_z (2, n_gp, n_ring), lumen volume (x vw), section centres, ring
        directions, section radii)"""
        sc = jnp.array([1.0, L0, 1.0, L0])
        E = jnp.stack([z[0:3], z[3:6], z[6:9], z[9:12]])
        a, A = z[12] * (1 - XG) + z[13] * XG, A_end[0] * (1 - XG) + A_end[1] * XG
        da, dA = (z[13] - z[12]) / L0, (A_end[1] - A_end[0]) / L0
        x, d1, d2 = (jnp.asarray(Hk) * sc / L0 ** k @ E for k, Hk in enumerate((H0, H1, H2)))
        lam = jnp.linalg.norm(d1, axis=-1)
        t = d1 / lam[:, None]
        tp = (d2 - jnp.sum(t * d2, -1, keepdims=True) * t) / lam[:, None]
        c, cos = jnp.cross(T0, t), t @ T0
        e0 = (self.cos_th[:, None] * D1 + self.sin_th[:, None] * D2)[None]
        cc = c[:, None]
        e = e0 + jnp.cross(cc, e0) + jnp.cross(cc, jnp.cross(cc, e0)) / (1 + cos)[:, None, None]
        lz = jnp.sqrt((lam[:, None] - a[:, None] * jnp.sum(e * tp[:, None], -1)) ** 2 + da ** 2) / jnp.sqrt(1 + dA ** 2)
        eps = jnp.stack([jnp.broadcast_to(jnp.log(a / A)[:, None], lz.shape), jnp.log(lz)])
        return eps, vw * jnp.sum(WG * L0 * jnp.pi * a ** 2 * lam), x, e, a

    def _deformation(self, eps, Jg):
        lt, lz = jnp.exp(eps[0]), jnp.exp(eps[1])
        return jnp.stack([Jg / (lt * lz), lt, lz], -1)[..., None] * jnp.eye(3)

    def _wall(self, st, F, nd):
        """(sigma_tt, sigma_zz) with sigma_rr = 0 (nd: point dimensions)"""
        f = self.model.sigma_solver
        for _ in range(nd):
            f = jax.vmap(f)
        s = f(st, F)[0]
        return s[..., 1, 1] - s[..., 0, 0], s[..., 2, 2] - s[..., 0, 0]

    def _residual_e(self, z, geo, st, dV0, p):
        eps, vjp = jax.vjp(lambda z: self._element(z, *geo)[0], z)
        Jg = jax.vmap(jax.vmap(self.model.J_target))(st)
        s = Jg * dV0 * jnp.stack(self._wall(st, self._deformation(eps, Jg), 2))
        return vjp(s)[0] - p * jax.grad(lambda z: self._element(z, *geo)[1])(z)

    def _jkin(self, zs, pd):
        """junction point: in-plane deformation (F_tt, F_tz, F_zz) in the current basis (e_1 = g_t / |g_t|, e_2 =
        n x e_1) and its lumen volume share 1/3 x . n dA"""
        L0, T0, X0, Ae, X, Eth, Ez, w, _, dA = pd
        phi = lambda Y: sum(w[k] * section_map(zs[k], L0[k], T0[k], X0[k], Ae[k], Y) for k in range(zs.shape[0]))
        x, G = phi(X), jax.jacfwd(phi)(X)
        gt, gz = G @ Eth, G @ Ez
        n = jnp.cross(gt, gz)
        e1 = gt / jnp.linalg.norm(gt)
        e2 = jnp.cross(n / jnp.linalg.norm(n), e1)
        return jnp.array([e1 @ gt, e1 @ gz, e2 @ gz]), x @ n * dA / 3

    def _jF(self, F2, Jg):
        return jnp.array([[Jg / (F2[0] * F2[2]), 0.0, 0.0], [0.0, F2[0], F2[1]], [0.0, 0.0, F2[2]]])

    def _jres(self, zs, pd, st, p):
        """junction point residual on its elements: J_g dV_0 (sigma - sigma_rr I) F^-T : dF - p dV"""
        Jg = self.model.J_target(st)
        (F2, _), vjp = jax.vjp(lambda zs: self._jkin(zs, pd), zs)
        F = self._jF(F2, Jg)
        s = self.model.sigma_solver(st, F)[0]
        P = Jg * (s - s[0, 0] * jnp.eye(3)) @ jnp.linalg.inv(F).T
        return vjp((pd[8] * jnp.array([P[1, 1], P[1, 2], P[2, 2]]), -p))[0]

    def _disc(self, z, k, s):
        """1/3 x . n pi a^2 of the end section k of an element, n = s t / |t|"""
        x, t, a = jnp.where(k == 0, z[0:3], z[6:9]), jnp.where(k == 0, z[3:6], z[9:12]), jnp.where(k == 0, z[12], z[13])
        return x @ (s * t / jnp.linalg.norm(t)) * jnp.pi * a ** 2 / 3

    def _rz_impl(self, z, st, p):
        rz = self._res_e(z, self.geo, self._per_el(st), self.dV0, p)
        if self.junction is not None:
            J = self.junction
            stj = self._flat(st, slice(self.n_rp, None))
            rz = rz.at[J["elems"]].add(jax.vmap(self._jres, (0, 0, 0, None))(z[J["elems"]], J["pd"], stj, p))
            rz = rz.at[J["disc_el"]].add(-p * jax.vmap(jax.grad(self._disc))(z[J["disc_el"]], J["disc_k"], J["disc_s"]))
        return rz

    def strains(self, q):
        eps = jax.vmap(self._element)(self.expand(q), *self.geo)[0]
        return jnp.moveaxis(eps, 1, 0).reshape(2, -1)

    def _F(self, q, st):
        """(n_points, 1, 3, 3)"""
        Jg = jax.vmap(self.model.J_target)(self._flat(st, slice(None)))
        F = self._deformation(self.strains(q), Jg[:self.n_rp])
        if self.junction is not None:
            J = self.junction
            F2 = jax.vmap(lambda zs, pd: self._jkin(zs, pd)[0])(self.expand(q)[J["elems"]], J["pd"])
            F = jnp.concatenate([F, jax.vmap(self._jF)(F2, Jg[self.n_rp:])])
        return F[:, None]

    def map_points(self, q, X, elems, w):
        """current positions Phi(X) = sum_k w_k section_map(z_k, X) of reference points X (n, 3) moved by the elements
        elems (n, k) with weights w (n, k), as the junction points"""
        L0, T0, _, _, A_end, _ = self.geo
        el = jnp.asarray(elems)
        phi = lambda zs, l, t, x0, ae, wk, Y: sum(wk[k] * section_map(zs[k], l[k], t[k], x0[k], ae[k], Y)
                                                  for k in range(zs.shape[0]))
        return np.asarray(jax.vmap(phi)(self._expand(q)[el], L0[el], T0[el], jnp.asarray(self.Xe[:, 0])[el],
                                        A_end[el], jnp.asarray(w, float), jnp.asarray(X, float)))

    def tangent(self, q, st, p):
        """K = J^T blockdiag(K_e) J + r_e . d2 z / dq2,  z = expand(q); z is linear in all DOFs but the junction
        rotations and end stretches (nl): J = J_0 + their columns, the second derivative only in their block;
        junction points couple the elements of their tubes (blocks of 3 x 14 entries of z)"""
        z, stl = self._expand(q), self._per_el(st)
        rz = self._rz(z, st, p)
        Kz = np.asarray(self._K_e(z, self.geo, stl, self.dV0, p)).ravel()
        Jn = np.asarray(self._J_nl(q)).reshape(self._J0.shape[0], -1)
        r, c = np.nonzero(Jn)
        J = self._J0 + sp.csr_matrix((Jn[r, c], (r, self.nl[c])), self._J0.shape)
        Kb = sp.csr_matrix((Kz, (self._rows, self._cols)), (J.shape[0], J.shape[0]))
        if self.junction is not None:
            Jc = self.junction
            for blocks, els in ((self._KJ(z, st, p), np.asarray(Jc["elems"])),
                                (self._Kd(z, p), np.asarray(Jc["disc_el"])[:, None])):
                ix = (els[..., None] * 14 + np.arange(14)).reshape(len(els), -1)
                m = ix.shape[1]
                Kb = Kb + sp.csr_matrix((np.asarray(blocks).reshape(len(els), m, m).ravel(),
                                         (np.repeat(ix, m, 1).ravel(), np.tile(ix, (1, m)).ravel())), Kb.shape)
        Gn = np.asarray(self._G_nl(q, rz))
        G = sp.csr_matrix((Gn.ravel(), (np.repeat(self.nl, len(self.nl)), np.tile(self.nl, len(self.nl)))),
                          (len(q), len(q)))
        if len(self.ties):
            Kt = np.asarray(self._tie_K(jnp.asarray(q)[self._tie_q], self.tie_L0, self.tie_Lr, self.tie_EA))
            G = G + sp.csr_matrix((Kt.ravel(), (np.repeat(self._tie_q, 6, 1).ravel(), np.tile(self._tie_q, 6).ravel())),
                                  (len(q), len(q)))
        return (J.T @ Kb @ J + G).tocsc()

    def solve(self, q, st, p, tol=1e-10):
        """Newton (steps below 0.1 min A); if it fails, pseudo-transient continuation as fem.System._ptc:
        (K + eta |diag K|) dq = -r, accepted unless |r| grows tenfold (eta x0.3), else eta x10 -> (q, iterations)"""
        x, step = q, 0.1 * float(np.min(self.geo[4]))
        for it in range(1, 31):
            dq = spla.spsolve(self.tangent(x, st, p), -np.asarray(self.residual(x, st, p)))
            if not np.isfinite(dq).all() or np.abs(dq).max() > step:
                break
            x = x + dq
            if np.abs(dq).max() < tol:
                return x, it
        r, eta = np.asarray(self.residual(q, st, p)), 1.0
        for it in range(1, 2001):
            K = self.tangent(q, st, p)
            dq = spla.spsolve(K + eta * sp.diags(np.abs(K.diagonal())), -r)
            rn = np.asarray(self.residual(q + dq, st, p))
            if not (np.isfinite(rn).all() and np.abs(rn).max() < 10 * np.abs(r).max()):
                eta *= 10
                if eta > 1e12:
                    break
                continue
            q, r, eta = q + dq, rn, 0.3 * eta
            if eta < 1e-10 and np.abs(dq).max() < tol:
                return q, 30 + it
        raise RuntimeError(f"rod relaxation did not converge, |r| = {np.abs(r).max():.2e}")

    def prestress(self, make, G_e, p, pre_iter=200, anderson=2, tol=1e-8):
        """G_e <- F G_e and tie rest lengths L_r <- L_r - (l - L_0) until the equilibrium at p is the reference (as the
        fem prestress; points of inactive elements keep G_e); make(G_e) -> states; -> (states, iterations,
        max|q - q_0|)"""
        xs, fs = [], []
        n_G = G_e.size
        q = self.q0
        live = jnp.asarray(np.r_[np.repeat(self.active, self.n_rp // self.n_el),
                                 np.ones(len(self.X_points) - self.n_rp, bool)])[:, None, None, None]
        for it in range(1, pre_iter + 1):
            st = make(G_e)
            q, _ = self.solve(q, st, p)
            du = float(np.abs(q - self.q0).max())
            if du < tol:
                break
            y = np.asarray(q)[self._tie_q] if len(self.ties) else np.zeros((0, 6))
            x = jnp.concatenate([G_e.ravel(), jnp.asarray(self.tie_Lr)])
            f = jnp.concatenate([(live * (self.F(q, st) @ G_e - G_e)).ravel(),
                                 jnp.asarray(self.tie_L0 - np.linalg.norm(y[:, 3:] - y[:, :3], axis=1))])
            xs, fs = (xs + [x])[-anderson - 1:], (fs + [f])[-anderson - 1:]
            x = fem.anderson(xs, fs)
            G_e, self.tie_Lr = x[:n_G].reshape(G_e.shape), np.asarray(x[n_G:])
        return st, it, du

    def nonlocal_mean(self, v, length):
        """v_i = sum_j w_ij dV_j v_j / sum_j w_ij dV_j,  w_ij = exp(-|X_i - X_j|^2 / length^2) within 2.5 length
        (reference material points, ring and junction)"""
        if length not in self._nl:
            P = cKDTree(self.X_points)
            W = P.sparse_distance_matrix(P, 2.5 * length, output_type="coo_matrix")
            W = sp.coo_matrix((np.exp(-W.data ** 2 / length ** 2), (W.row, W.col)), W.shape).tolil()
            W.setdiag(1.0)
            W = W.tocsr() @ sp.diags(self.dV_points)
            self._nl[length] = sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
        return jnp.asarray((self._nl[length] @ np.asarray(v).ravel()).reshape(np.shape(v)))

    def wall(self, q, st):
        """hex8 wall for ParaView: points (inner / outer surface at a -+ h/2 around the deformed centerline,
        h = H lambda_r), connectivity between consecutive sections of a branch with wall (active elements)"""
        ne, ng, nr = self.n_el, len(XG), self.n_ring
        z = self._expand(q)
        _, _, x, e, a = jax.vmap(self._element)(z, *self.geo)
        h = self.H[..., None] * np.asarray(self.F(q, st))[:self.n_rp, 0, 0, 0].reshape(ne, ng, nr)
        rho = np.asarray(a)[..., None, None] + np.array([-0.5, 0.5]) * h[..., None]
        X = (np.asarray(x)[:, :, None, None] + rho[..., None] * np.asarray(e)[:, :, :, None]).reshape(-1, 3)
        sec, act = np.repeat(self.branch, ng), np.repeat(self.active, ng)
        nid = lambda k, j, l: (k * nr + j % nr) * 2 + l
        conn = np.array([[nid(k, j + 1, 0), nid(k, j, 0), nid(k, j, 1), nid(k, j + 1, 1),
                          nid(k + 1, j + 1, 0), nid(k + 1, j, 0), nid(k + 1, j, 1), nid(k + 1, j + 1, 1)]
                         for k in range(ne * ng - 1) if sec[k] == sec[k + 1] and act[k] and act[k + 1]
                         for j in range(nr)])
        return X, conn
