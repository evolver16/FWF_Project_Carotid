"""Reduced artery model: network of geometrically exact C1 rods (cubic Hermite centerline in 3D), each section a thin
ring of material points of a fem-style G&R model (sigma_solver, J_target, commit), branches joined at rigid junctions."""

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
    the pressure load p pi a^2 of a curved tube)."""

    def __init__(self, X, conn, branch, A, H, model, rigid=(), clamped=(), n_ring=16):
        self.X, self.conn, self.branch, self.model = np.asarray(X, float), np.asarray(conn), np.asarray(branch), model
        n, ne, ng = len(self.X), len(self.conn), len(XG)
        self.n_el, self.n_ring = ne, n_ring
        d = self.X[self.conn[:, 1]] - self.X[self.conn[:, 0]]
        L0 = np.linalg.norm(d, axis=1)
        T0 = d / L0[:, None]
        D1 = np.cross([0.0, 1.0, 0.0], T0)
        D1 /= np.linalg.norm(D1, axis=1, keepdims=True)
        D2 = np.cross(T0, D1)
        th = (np.arange(n_ring) + 0.5) * 2 * np.pi / n_ring
        self.th, self.cos_th, self.sin_th = th, jnp.asarray(np.cos(th)), jnp.asarray(np.sin(th))
        A_end = np.asarray(A, float)
        self.A, self.H = A_end[:, :1] * (1 - XG) + A_end[:, 1:] * XG, np.asarray(H, float)
        self.geo = tuple(jnp.asarray(v) for v in (L0, T0, D1, D2, A_end))
        self.dV0 = jnp.asarray(np.repeat((WG * L0[:, None] * self.A * self.H * 2 * np.pi / n_ring)[..., None], n_ring, -1))
        ring = np.cos(th)[None, :, None] * D1[:, None] + np.sin(th)[None, :, None] * D2[:, None]
        xs = self.X[self.conn[:, 0]][:, None] + XG[None, :, None] * d[:, None]
        self.X_points = (xs[:, :, None] + self.A[..., None, None] * ring[:, None]).reshape(-1, 3)
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
        self._nl = {}
        nz = 14
        self._rows = (np.arange(ne)[:, None, None] * nz + np.arange(nz)[:, None] + 0 * np.arange(nz)).ravel()
        self._cols = (np.arange(ne)[:, None, None] * nz + 0 * np.arange(nz)[:, None] + np.arange(nz)).ravel()
        self._per_el = lambda st: jax.tree.map(lambda x: x.reshape(ne, ng, *x.shape[1:]), st)
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
        self._residual = jax.jit(lambda q, st, p: jax.vjp(self.expand, q)[1](
            self._res_e(self.expand(q), self.geo, self._per_el(st), self.dV0, p))[0])
        self.ties, self.tie_L0, self.tie_Lr, self._tie_q = np.zeros((0, 2), int), np.zeros(0), np.zeros(0), None
        bar = lambda y, L0, Lr, EA: 0.5 * EA * (jnp.linalg.norm(y[3:] - y[:3]) - Lr) ** 2 / L0
        self._tie_g = jax.jit(jax.vmap(jax.grad(bar)))
        self._tie_K = jax.jit(jax.vmap(jax.hessian(bar)))
        self.F = jax.jit(self._F)
        self.stresses = jax.jit(lambda q, st: self._wall(st, self._F(q, st)))
        self.commit = jax.jit(lambda q, st: (lambda F: jax.vmap(jax.vmap(model.commit))(
            st, F, jax.vmap(jax.vmap(model.sigma_solver))(st, F)[1]))(self._F(q, st)))

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
            t = jnp.where(self.rigid[nd][:, None], lam[:, k, None] * jnp.einsum("eij,ej->ei", Rn[nd], self.geo[1]),
                          tau[nd])
            out += [r[nd], t]
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

    def _element(self, z, L0, T0, D1, D2, A_end):
        """(strains ln lambda_theta, ln lambda_z (2, n_gp, n_ring), lumen volume, section centres, ring directions,
        section radii)"""
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
        return eps, jnp.sum(WG * L0 * jnp.pi * a ** 2 * lam), x, e, a

    def _deformation(self, eps, st):
        lt, lz = jnp.exp(eps[0]), jnp.exp(eps[1])
        Jg = jax.vmap(jax.vmap(self.model.J_target))(st)
        return jnp.stack([Jg / (lt * lz), lt, lz], -1)[..., None] * jnp.eye(3)

    def _wall(self, st, F):
        """(sigma_tt, sigma_zz) with sigma_rr = 0"""
        s = jax.vmap(jax.vmap(self.model.sigma_solver))(st, F)[0]
        return s[..., 1, 1] - s[..., 0, 0], s[..., 2, 2] - s[..., 0, 0]

    def _residual_e(self, z, geo, st, dV0, p):
        eps, vjp = jax.vjp(lambda z: self._element(z, *geo)[0], z)
        Jg = jax.vmap(jax.vmap(self.model.J_target))(st)
        s = Jg * dV0 * jnp.stack(self._wall(st, self._deformation(eps, st)))
        return vjp(s)[0] - p * jax.grad(lambda z: self._element(z, *geo)[1])(z)

    def strains(self, q):
        eps = jax.vmap(self._element)(self.expand(q), *self.geo)[0]
        return jnp.moveaxis(eps, 1, 0).reshape(2, -1, self.n_ring)

    def _F(self, q, st):
        return self._deformation(self.strains(q), st)

    def tangent(self, q, st, p):
        """K = J^T blockdiag(K_e) J + r_e . d2 z / dq2,  z = expand(q); z is linear in all DOFs but the junction
        rotations and end stretches (nl): J = J_0 + their columns, the second derivative only in their block"""
        z, stl = self._expand(q), self._per_el(st)
        rz = self._res_e(z, self.geo, stl, self.dV0, p)
        Kz = np.asarray(self._K_e(z, self.geo, stl, self.dV0, p)).ravel()
        Jn = np.asarray(self._J_nl(q)).reshape(self._J0.shape[0], -1)
        r, c = np.nonzero(Jn)
        J = self._J0 + sp.csr_matrix((Jn[r, c], (r, self.nl[c])), self._J0.shape)
        Kb = sp.csr_matrix((Kz, (self._rows, self._cols)), (J.shape[0], J.shape[0]))
        Gn = np.asarray(self._G_nl(q, rz))
        G = sp.csr_matrix((Gn.ravel(), (np.repeat(self.nl, len(self.nl)), np.tile(self.nl, len(self.nl)))),
                          (len(q), len(q)))
        if len(self.ties):
            Kt = np.asarray(self._tie_K(jnp.asarray(q)[self._tie_q], self.tie_L0, self.tie_Lr, self.tie_EA))
            G = G + sp.csr_matrix((Kt.ravel(), (np.repeat(self._tie_q, 6, 1).ravel(), np.tile(self._tie_q, 6).ravel())),
                                  (len(q), len(q)))
        return (J.T @ Kb @ J + G).tocsc()

    def solve(self, q, st, p, tol=1e-10):
        """Newton; if it fails, pseudo-transient continuation as fem.System._ptc: (K + eta |diag K|) dq = -r,
        accepted unless |r| grows tenfold (eta x0.3), else eta x10 -> (q, iterations)"""
        x = q
        for it in range(1, 31):
            dq = spla.spsolve(self.tangent(x, st, p), -np.asarray(self.residual(x, st, p)))
            if not np.isfinite(dq).all():
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
        fem prestress); make(G_e) -> states; -> (states, iterations, max|q - q_0|)"""
        xs, fs = [], []
        n_G = G_e.size
        for it in range(1, pre_iter + 1):
            st = make(G_e)
            q, _ = self.solve(self.q0, st, p)
            du = float(np.abs(q - self.q0).max())
            if du < tol:
                break
            y = np.asarray(q)[self._tie_q] if len(self.ties) else np.zeros((0, 6))
            x = jnp.concatenate([G_e.ravel(), jnp.asarray(self.tie_Lr)])
            f = jnp.concatenate([(self.F(q, st) @ G_e - G_e).ravel(),
                                 jnp.asarray(self.tie_L0 - np.linalg.norm(y[:, 3:] - y[:, :3], axis=1))])
            xs, fs = (xs + [x])[-anderson - 1:], (fs + [f])[-anderson - 1:]
            x = fem.anderson(xs, fs)
            G_e, self.tie_Lr = x[:n_G].reshape(G_e.shape), np.asarray(x[n_G:])
        return st, it, du

    def nonlocal_mean(self, v, length):
        """v_i = sum_j w_ij dV_j v_j / sum_j w_ij dV_j,  w_ij = exp(-|X_i - X_j|^2 / length^2) within 2.5 length
        (reference ring points)"""
        if length not in self._nl:
            P = cKDTree(self.X_points)
            W = P.sparse_distance_matrix(P, 2.5 * length, output_type="coo_matrix")
            W = sp.coo_matrix((np.exp(-W.data ** 2 / length ** 2), (W.row, W.col)), W.shape).tolil()
            W.setdiag(1.0)
            W = W.tocsr() @ sp.diags(np.asarray(self.dV0).ravel())
            self._nl[length] = sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
        return jnp.asarray((self._nl[length] @ np.asarray(v).ravel()).reshape(np.shape(v)))

    def wall(self, q, st):
        """hex8 wall for ParaView: points (inner / outer surface at a -+ h/2 around the deformed centerline,
        h = H lambda_r), connectivity between consecutive sections of a branch"""
        ne, ng, nr = self.n_el, len(XG), self.n_ring
        z = self._expand(q)
        _, _, x, e, a = jax.vmap(self._element)(z, *self.geo)
        h = self.H[..., None] * np.asarray(self.F(q, st))[..., 0, 0].reshape(ne, ng, nr)
        rho = np.asarray(a)[..., None, None] + np.array([-0.5, 0.5]) * h[..., None]
        X = (np.asarray(x)[:, :, None, None] + rho[..., None] * np.asarray(e)[:, :, :, None]).reshape(-1, 3)
        sec = np.repeat(self.branch, ng)
        nid = lambda k, j, l: (k * nr + j % nr) * 2 + l
        conn = np.array([[nid(k, j + 1, 0), nid(k, j, 0), nid(k, j, 1), nid(k, j + 1, 1),
                          nid(k + 1, j + 1, 0), nid(k + 1, j, 0), nid(k + 1, j, 1), nid(k + 1, j + 1, 1)]
                         for k in range(ne * ng - 1) if sec[k] == sec[k + 1] for j in range(nr)])
        return X, conn
