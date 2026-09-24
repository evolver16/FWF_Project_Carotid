"""Total-Lagrangian hex8 FE: kernels, assembly, follower pressure, Newton, G&R time loop.

Model interface per integration point (hcmm, fcmm, elastic):
    sigma, aux = model.sigma_solver(state, F)
    state      = model.commit(state, F, aux)
    J_target(state)                              optional, hybrid element (default 1)
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import setups  # noqa: F401  (x64, compilation cache)
from elements import element_for, face_for
from tensor3 import det3, inv3

ELEMENTS = ("standard", "fbar", "hybrid")


def geometry(X, conn):
    """dN/dX and w det(dX/dxi) at the Gauss points, (n_el, n_gp, n_en, 3), (n_el, n_gp)"""
    el = element_for(conn.shape[1])
    dNdxi = np.stack([el.dN(g) for g in el.gauss])
    J0 = np.einsum("eai,gaj->egij", X[conn], dNdxi)
    return np.einsum("gaj,egji->egai", dNdxi, np.linalg.inv(J0)), el.weights * np.linalg.det(J0)


def centroid_dNdX(X, conn):
    """dN/dX at the element centre, (n_el, n_en, 3)"""
    el = element_for(conn.shape[1])
    dNdxi = el.dN(el.centre)
    J0 = np.einsum("eai,aj->eij", X[conn], dNdxi)
    return np.einsum("aj,eji->eai", dNdxi, np.linalg.inv(J0))


def gauss_points(X, conn):
    """Reference coordinates of the Gauss points, (n_el, n_gp, 3)"""
    el = element_for(conn.shape[1])
    return np.einsum("ga,eai->egi", np.stack([el.N(g) for g in el.gauss]), X[conn])


def laplace(X, conn, fixed):
    """Nodal phi with div grad phi = 0, phi = v on each (node set, v) in fixed, zero flux elsewhere"""
    dNdX, wdet = geometry(X, conn)
    Ke = np.einsum("eg,egai,egbi->eab", wdet, dNdX, dNdX)
    n = len(X)
    nen = conn.shape[1]
    K = sp.csr_matrix((Ke.ravel(), (np.repeat(conn, nen, axis=1).ravel(), np.tile(conn, nen).ravel())),
                      shape=(n, n))
    phi, on = np.zeros(n), np.zeros(n, dtype=bool)
    for idx, v in fixed:
        phi[idx], on[idx] = v, True
    phi[~on] = spla.spsolve(K[~on][:, ~on].tocsc(), -K[~on][:, on] @ phi[on])
    return phi


def wall_basis(X, conn, inner, outer, inlets, outlets):
    """Columns (e_r, e_theta, e_z) at the Gauss points, (n_el, n_gp, 3, 3), from two Laplace fields:
    e_r ~ grad phi_r (inner 0 -> outer 1),  e_z ~ grad phi_z (inlets 0 -> outlets 1) minus its e_r part,
    e_theta = e_z x e_r
    """
    dNdX, _ = geometry(X, conn)
    grad = lambda phi: np.einsum("egai,ea->egi", dNdX, phi[conn])
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    e_r = unit(grad(laplace(X, conn, [(inner, 0.0), (outer, 1.0)])))
    g_z = grad(laplace(X, conn, [(i, 0.0) for i in inlets] + [(o, 1.0) for o in outlets]))
    e_z = unit(g_z - np.sum(g_z * e_r, axis=-1, keepdims=True) * e_r)
    return np.stack([e_r, np.cross(e_z, e_r), e_z], axis=-1)


def anderson(xs, fs, eps=1e-12):
    """Anderson (type II) step from iterates x_k and residuals f_k = g(x_k) - x_k:
    x+ = x + f - (dX + dF) gamma,  (dF^T dF + eps tr(dF^T dF) I) gamma = dF^T f
    """
    x, f = xs[-1], fs[-1]
    if len(xs) == 1:
        return x + f
    dX = jnp.stack([b - a for a, b in zip(xs[:-1], xs[1:])], 1)
    dF = jnp.stack([b - a for a, b in zip(fs[:-1], fs[1:])], 1)
    A = dF.T @ dF
    gamma = jnp.linalg.solve(A + eps * jnp.trace(A) * jnp.eye(A.shape[0]), dF.T @ f)
    return x + f - (dX + dF) @ gamma


def broadcast_state(state, n_el, n_gp=8):
    """One material state copied to every integration point, leaves (n_el, n_gp, ...)"""
    return jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x), (n_el, n_gp) + jnp.shape(x)), state)


def save_state(path, states, u):
    """Restart file: leaves of the state pytree and u (.npz)"""
    np.savez(path, *[np.asarray(x) for x in jax.tree.leaves(states)], u=np.asarray(u))


def load_state(path, like):
    """(states, u) from save_state; like: any state pytree of the same structure (e.g. the initial one)"""
    with np.load(path) as f:
        leaves = [jnp.asarray(f[f"arr_{i}"]) for i in range(len(f.files) - 1)]
        return jax.tree.unflatten(jax.tree.structure(like), leaves), jnp.asarray(f["u"])


@dataclass
class BC:
    """Dirichlet dofs/values, dead nodal forces, follower pressure p on the pressure faces,
    slip = (nodes, normals): u . n = 0 at those nodes (inclined supports, symmetry planes)."""
    fixed: np.ndarray
    values: np.ndarray
    f_dead: np.ndarray = None
    p: float = 0.0
    slip: tuple = None


class Constraints:
    """x = x_p + Z y: Z selects the free dofs, or with slip conditions spans per node the null space of
    the node's constraint normals C (rows e_k for fixed dofs, n for slip), x_p = C^+ v"""

    def __init__(self, n_x, n_dof, bc):
        self.n_x = n_x
        fixed = np.asarray(bc.fixed, dtype=int)
        values = np.asarray(bc.values, dtype=float)
        self.x_p = np.zeros(n_x)
        self.x_p[fixed] = values
        self.free, self.Z = None, None
        if bc.slip is None:
            self.free = np.ones(n_x, dtype=bool)
            self.free[fixed] = False
            return
        cons = {}
        for dof, v in zip(fixed, values):
            cons.setdefault(dof // 3, []).append((np.eye(3)[dof % 3], v))
        for i, n in zip(*bc.slip):
            cons.setdefault(int(i), []).append((np.asarray(n, dtype=float) / np.linalg.norm(n), 0.0))
        open_nodes = np.setdiff1d(np.arange(n_dof // 3), list(cons))
        rows = [(3 * open_nodes[:, None] + np.arange(3)).ravel(), np.arange(n_dof, n_x)]
        vals = [np.ones(3 * len(open_nodes) + n_x - n_dof)]
        for i, c in cons.items():
            A, v = np.array([a for a, _ in c]), np.array([b for _, b in c])
            _, S, Vt = np.linalg.svd(A)
            null = Vt[int((S > 1e-10 * S[0]).sum()):]
            self.x_p[3 * i:3 * i + 3] = np.linalg.lstsq(A, v, rcond=None)[0]
            rows.append(np.tile(3 * i + np.arange(3), len(null)))
            vals.append(null.ravel())
        rows = np.concatenate(rows)
        counts = np.concatenate([np.ones(3 * len(open_nodes) + n_x - n_dof, dtype=int),
                                 np.repeat(3, sum(len(v) for v in vals[1:]) // 3)])
        cols = np.repeat(np.arange(len(counts)), counts)
        self.Z = sp.csc_matrix((np.concatenate(vals), (rows, cols)), shape=(n_x, len(counts)))

    def restrict(self, v):
        return v[self.free] if self.Z is None else self.Z.T @ v

    def prolong(self, y):
        if self.Z is not None:
            return self.Z @ y
        out = np.zeros(self.n_x)
        out[self.free] = y
        return out

    def project(self, x):
        return self.x_p + self.prolong(self.restrict(x - self.x_p))

    def restrict_scale(self, s):
        return s[self.free] if self.Z is None else (abs(self.Z).T @ s) / (abs(self.Z).T @ np.ones(self.n_x))


def gmres(A, b, M, tol, m):
    """Right-preconditioned GMRES, min_y |b - A M V y| over the Krylov basis V -> (x, iterations), x = None if not converged"""
    beta = np.linalg.norm(b)
    if beta == 0:
        return np.zeros_like(b), 0
    V, H = np.zeros((m + 1, b.size)), np.zeros((m + 1, m))
    V[0] = b / beta
    e1 = np.zeros(m + 1)
    e1[0] = beta
    for j in range(m):
        w = A @ M(V[j])
        for i in range(j + 1):
            H[i, j] = V[i] @ w
            w -= H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(w)
        y = np.linalg.lstsq(H[:j + 2, :j + 1], e1[:j + 2], rcond=None)[0]
        if np.linalg.norm(H[:j + 2, :j + 1] @ y - e1[:j + 2]) <= tol * beta or H[j + 1, j] == 0:
            return M(V[:j + 1].T @ y), j + 1
        V[j + 1] = w / H[j + 1, j]
    return None, m


class LinearSolver:
    """K x = b or K^T x = b by GMRES preconditioned with the last LU of K; refactorized when that fails."""

    def __init__(self, tol=1e-12, max_iter=20, refactor_after=5, ordering="MMD_AT_PLUS_A"):
        self.tol, self.max_iter, self.refactor_after, self.ordering = tol, max_iter, refactor_after, ordering
        self.lu = None
        self.factorizations = self.iterations = 0

    def __call__(self, K, b, trans="N"):
        if not (np.isfinite(K.data).all() and np.isfinite(b).all()):
            raise RuntimeError("linear solve: non-finite tangent or right-hand side")
        A = K if trans == "N" else K.T
        if self.lu is not None and self.lu.shape == K.shape:
            x, it = gmres(A, b, lambda v: self.lu.solve(v, trans=trans), self.tol, self.max_iter)
            self.iterations += it
            if x is not None and np.linalg.norm(A @ x - b) <= 10 * self.tol * np.linalg.norm(b):
                if it > self.refactor_after:
                    self.lu = None
                return x
        self.lu = spla.splu(K.tocsc(), permc_spec=self.ordering)
        self.factorizations += 1
        return self.lu.solve(b, trans=trans)


class System:
    """Mesh of hex8 (2x2x2 Gauss) or tet10 (4-point); element formulation:
    "standard", "fbar" (F^ = (J_0/J)^(1/3) F, J_0 at the element centre),
    "hybrid" (Q1/P0 or P2/P0: sigma - q I with one pressure q per element and J = J_target on average)

    fbar develops hourglass-type modes in long G&R runs for K/mu >~ 1e3; use hybrid there.
    """

    def __init__(self, mesh, model, pressure_faces=None, batch=128, element="standard"):
        if element not in ELEMENTS:
            raise ValueError(f"element must be one of {ELEMENTS}, got {element!r}")
        self.mesh, self.model, self.batch, self.element = mesh, model, batch, element
        self.n_dof = 3 * mesh.n_nodes
        self.n_p = mesh.n_elem if element == "hybrid" else 0
        self.n_x = self.n_dof + self.n_p
        dNdX, wdet = geometry(mesh.X, mesh.conn)
        bad = np.flatnonzero((wdet <= 0).any(axis=1))
        if len(bad):
            raise ValueError(f"{len(bad)} elements with det(dX/dxi) <= 0 at a Gauss point "
                             f"(inverted or misordered), e.g. {bad[:5]}")
        self.dNdX, self.wdet = jnp.asarray(dNdX), jnp.asarray(wdet)
        self.dNdX0 = jnp.asarray(centroid_dNdX(mesh.X, mesh.conn))
        self._p_scale = wdet.sum(axis=1)[:self.n_p]
        self.J_target = getattr(model, "J_target", lambda s: jnp.ones(()))
        self.conn = jnp.asarray(mesh.conn)
        self.n_en, self.n_gp = mesh.conn.shape[1], wdet.shape[1]
        faces = np.zeros((0, 4 if self.n_en == 8 else 6), dtype=int) if pressure_faces is None else pressure_faces
        self.faces = jnp.asarray(faces)
        self.X = jnp.asarray(mesh.X)
        self.length = float(np.ptp(mesh.X, axis=0).max())
        face = face_for(faces.shape[1])
        self.N2 = jnp.asarray(np.stack([face.N(g) for g in face.gauss]))
        self.dN2 = jnp.asarray(np.stack([face.dN(g) for g in face.gauss]))
        self.w2 = jnp.asarray(face.weights)

        ne = 3 * self.n_en
        edofs = (3 * mesh.conn[:, :, None] + np.arange(3)).reshape(mesh.n_elem, ne)
        if self.n_p:
            edofs = np.hstack([edofs, self.n_dof + np.arange(mesh.n_elem)[:, None]])
        nd, nf = edofs.shape[1], 3 * faces.shape[1]
        fdofs = (3 * np.asarray(faces)[:, :, None] + np.arange(3)).reshape(-1, nf)
        self._rows = np.concatenate([np.repeat(edofs, nd, axis=1).ravel(),
                                     np.repeat(fdofs, nf, axis=1).ravel()])
        self._cols = np.concatenate([np.tile(edofs, nd).ravel(), np.tile(fdofs, nf).ravel()])

        self.residual = jax.jit(self._residual)
        self.tangent = jax.jit(self._tangent)
        self.gauss_F = jax.jit(self._gauss_F)
        self.stress = jax.jit(self._stress)
        self._commit_jit = jax.jit(jax.checkpoint(self._commit))
        self.residual_vjp = jax.jit(self._residual_vjp)
        self.linear = LinearSolver(ordering="COLAMD" if self.n_p else "MMD_AT_PLUS_A")
        self.last_u, self.last_iterations = None, 0
        self.last_p = np.zeros(self.n_p)
        self._bc_last, self.cutbacks = None, 0
        self._patterns, self._passthrough, self._cons = {}, {}, {}

    def _Fhat(self, F, F0):
        """Deformation gradient seen by the material: F-bar (J_0/J)^(1/3) F, else F"""
        if self.element != "fbar":
            return F
        return (det3(F0) / det3(F))[..., None, None] ** (1.0 / 3.0) * F

    def to_nodes(self, values):
        """Gauss-point field (n_el, n_gp, ...) -> nodes (n_nodes, ...), volume-weighted element means"""
        v = np.asarray(values)
        w = np.asarray(self.wdet)
        mean = np.einsum("eg,eg...->e...", w, v) / w.sum(1).reshape((-1,) + (1,) * (v.ndim - 2))
        vol = np.repeat(w.sum(1), self.n_en)
        out = np.zeros((self.mesh.n_nodes,) + v.shape[2:])
        np.add.at(out, self.mesh.conn.ravel(), np.repeat(mean, self.n_en, axis=0) * vol.reshape((-1,) + (1,) * (v.ndim - 2)))
        acc = np.bincount(self.mesh.conn.ravel(), weights=vol, minlength=self.mesh.n_nodes)
        return out / acc.reshape((-1,) + (1,) * (v.ndim - 2))

    def _split(self, x):
        """x = (u, element pressures); zero pressures without the hybrid element"""
        return x[:self.n_dof], (x[self.n_dof:] if self.n_p else jnp.zeros(self.mesh.n_elem))

    def _gauss_F(self, u):
        """Material deformation gradient at every Gauss point, (n_el, n_gp, 3, 3)"""
        U = u.reshape(-1, 3)[self.conn]
        F = jnp.eye(3) + jnp.einsum("eai,egaj->egij", U, self.dNdX)
        F0 = jnp.eye(3) + jnp.einsum("eai,eaj->eij", U, self.dNdX0)
        return self._Fhat(F, F0[:, None])

    def _elem_F(self, u_e, dNdX_e, dNdX0_e):
        return (jnp.eye(3) + jnp.einsum("ai,gaj->gij", u_e, dNdX_e),
                jnp.eye(3) + jnp.einsum("ai,aj->ij", u_e, dNdX0_e))

    def _elem_force(self, u_e, q, st_e, dNdX_e, dNdX0_e, w_e):
        """f_ai = sum_g w_g P_iJ dN_a/dX_J,  P = J (sigma(F^) - q I) F^-T;  r_q = -sum_g w_g (J - J_target)"""
        F, F0 = self._elem_F(u_e, dNdX_e, dNdX0_e)
        sig = jax.vmap(lambda s, f: self.model.sigma_solver(s, f)[0])(st_e, self._Fhat(F, F0)) - q * jnp.eye(3)
        J = det3(F)
        P = J[:, None, None] * sig @ inv3(F).transpose(0, 2, 1)
        r_q = -jnp.sum(w_e * (J - jax.vmap(self.J_target)(st_e)))
        return jnp.einsum("g,gaJ,giJ->ai", w_e, dNdX_e, P), r_q

    def _face_force(self, x_f, p):
        """f_a = -p sum_g w_g N_a (dx/deta1 x dx/deta2)   (follower pressure on the deformed face)"""
        a1 = jnp.einsum("ai,ga->gi", x_f, self.dN2[:, :, 0])
        a2 = jnp.einsum("ai,ga->gi", x_f, self.dN2[:, :, 1])
        return -p * jnp.einsum("g,ga,gi->ai", self.w2, self.N2, jnp.cross(a1, a2))

    def _residual(self, x, states, p, f_dead):
        """r = (f_int(u, q) - f_pressure(u) - f_dead, r_q)"""
        u, q = self._split(x)
        U = u.reshape(-1, 3)
        fe, rq = jax.lax.map(lambda a: self._elem_force(*a),
                             (U[self.conn], q, states, self.dNdX, self.dNdX0, self.wdet), batch_size=self.batch)
        r = jnp.zeros_like(U).at[self.conn].add(fe)
        ff = jax.vmap(self._face_force, (0, None))((self.X + U)[self.faces], p)
        r = r.at[self.faces].add(-ff).ravel() - f_dead
        return jnp.concatenate([r, rq]) if self.n_p else r

    def _residual_vjp(self, x, states, p, f_dead, w):
        """(w . dr/dstates, w . dr/dp)"""
        return jax.vjp(lambda s, q: self._residual(x, s, q, f_dead), states, p)[1](w)

    def _elem_tangent(self, u_e, q, st_e, dNdX_e, dNdX0_e, w_e):
        """K_aibk = sum_g w_g dN_a/dX_J [(dP_iJ/dF_kL) dN_b/dX_L + (dP_iJ/dF0_kL) dN0_b/dX_L]
        hybrid: K_uq = -sum_g w_g J F^-T : dN = K_qu^T, K_qq = 0
        dsigma/dF^ by forward AD per Gauss point, then AD of P with sigma linearized about F^
        """
        F, F0 = self._elem_F(u_e, dNdX_e, dNdX0_e)
        fbar = self.element == "fbar"

        def gp(st, F):
            Fh = self._Fhat(F, F0)
            s = lambda f: self.model.sigma_solver(st, f)[0]
            sig, C = s(Fh), jax.jacfwd(s)(Fh)
            P = lambda F, F0: det3(F) * (sig + jnp.einsum("ijkl,kl->ij", C, self._Fhat(F, F0) - Fh)
                                         - q * jnp.eye(3)) @ inv3(F).T
            return jax.jacfwd(P, argnums=(0, 1) if fbar else 0)(F, F0)

        AB = jax.vmap(gp)(st_e, F)
        A = AB[0] if fbar else AB
        K = jnp.einsum("g,gaJ,giJkL,gbL->aibk", w_e, dNdX_e, A, dNdX_e)
        if fbar:
            K += jnp.einsum("g,gaJ,giJkL,bL->aibk", w_e, dNdX_e, AB[1], dNdX0_e)
        ne = 3 * self.n_en
        K = K.reshape(ne, ne)
        if not self.n_p:
            return K
        G = det3(F)[:, None, None] * inv3(F).transpose(0, 2, 1)
        kuq = -jnp.einsum("g,gaJ,giJ->ai", w_e, dNdX_e, G).reshape(ne, 1)
        return jnp.block([[K, kuq], [kuq.T, jnp.zeros((1, 1))]])

    def _tangent(self, x, states, p):
        """Element and face tangents dr/dx for sparse assembly"""
        u, q = self._split(x)
        U = u.reshape(-1, 3)
        ke = jax.lax.map(lambda a: self._elem_tangent(*a),
                         (U[self.conn], q, states, self.dNdX, self.dNdX0, self.wdet), batch_size=self.batch)
        nf = self.faces.shape[1]
        kf = jax.vmap(jax.jacfwd(lambda xf: -self._face_force(xf.reshape(nf, 3), p).ravel()))(
            (self.X + U)[self.faces].reshape(-1, 3 * nf))
        return jnp.concatenate([ke.ravel(), kf.ravel()])

    def _stress(self, u, states):
        """Material Cauchy stress at every Gauss point, (n_el, n_gp, 3, 3); hybrid total: minus last_p[e] I"""
        return jax.vmap(jax.vmap(lambda s, f: self.model.sigma_solver(s, f)[0]))(states, self._gauss_F(u))

    def _commit(self, u, states, commit=None):
        F = self._gauss_F(u)
        aux = jax.vmap(jax.vmap(lambda s, f: self.model.sigma_solver(s, f)[1]))(states, F)
        return jax.vmap(jax.vmap(commit or self.model.commit))(states, F, aux)

    def commit(self, u, states):
        """New states at every Gauss point; leaves the model passes through keep their input buffers."""
        leaves, tdef = jax.tree.flatten(states)
        if tdef not in self._passthrough:
            commit = getattr(self.model.commit, "__wrapped__", self.model.commit)
            spec = [jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)) for x in leaves]
            jx = jax.make_jaxpr(lambda s: self._commit(jnp.zeros(self.n_dof), tdef.unflatten(s), commit))(spec)
            ins = jx.jaxpr.invars
            self._passthrough[tdef] = [next((i for i, v in enumerate(ins) if v is o), None)
                                       for o in jx.jaxpr.outvars]
        new = jax.tree.leaves(self._commit_jit(u, states))
        return tdef.unflatten([n if i is None else leaves[i] for n, i in zip(new, self._passthrough[tdef])])

    def _constraints(self, bc):
        key = (np.asarray(bc.fixed).tobytes(), np.asarray(bc.values, dtype=float).tobytes(),
               None if bc.slip is None else tuple(np.asarray(a).tobytes() for a in bc.slip))
        if key not in self._cons:
            if len(self._cons) > 8:
                self._cons.clear()
            self._cons[key] = Constraints(self.n_x, self.n_dof, bc)
        return self._cons[key]

    def _K_red(self, x, states, p, cons):
        """Z^T K Z (selection: assembled directly into the free-dof pattern)"""
        if cons.Z is None:
            return self._K_free(x, states, p, cons.free)
        K = self._K_free(x, states, p, np.ones(self.n_x, dtype=bool))
        return (cons.Z.T @ K @ cons.Z).tocsc()

    def _f_dead(self, bc):
        return jnp.zeros(self.n_dof) if bc.f_dead is None else jnp.asarray(bc.f_dead)

    def _pattern(self, free):
        """CSC structure of K_ff and the map element entries -> nonzeros, cached per set of free dofs"""
        key = free.tobytes()
        if key not in self._patterns:
            idx = np.cumsum(free) - 1
            keep = np.flatnonzero(free[self._rows] & free[self._cols])
            n = int(free.sum())
            flat = idx[self._cols[keep]].astype(np.int64) * n + idx[self._rows[keep]]
            uniq, inv = np.unique(flat, return_inverse=True)
            indptr = np.searchsorted(uniq // n, np.arange(n + 1))
            self._patterns[key] = (keep, inv, (uniq % n, indptr), n)
        return self._patterns[key]

    def _K_free(self, x, states, p, free):
        keep, inv, (indices, indptr), n = self._pattern(free)
        vals = np.asarray(self.tangent(jnp.asarray(x), states, p))
        return sp.csc_matrix((np.bincount(inv, weights=vals[keep], minlength=len(indices)), indices, indptr),
                             shape=(n, n))

    def _solve(self, u0, states, bc, tol=1e-10, max_iter=30):
        """Newton on the free unknowns x = (u, element pressures) -> (x, n_iter)

        converged: |r_u| < tol max|r_u(u0)|, |r_q| < tol V_e, or |du| <= 1e-13 L
        failed Newton: load continuation from the last converged BC with cutback
        """
        try:
            x, it = self._newton(u0, states, bc, tol, max_iter)
        except RuntimeError:
            x, it = self._continuation(u0, states, bc, tol, max_iter)
        self._bc_last = bc
        return x, it

    def _continuation(self, u0, states, bc, tol, max_iter, min_step=2.0 ** -10):
        """Newton along bc(s) = (1 - s) bc_last + s bc, s: 0 -> 1, increment halved on failure, doubled on success"""
        start = self._bc_last
        same = lambda a, b: (a is None and b is None) or (a is not None and b is not None and all(
            np.array_equal(x, y) for x, y in zip(a, b)))
        if start is None or not np.array_equal(start.fixed, bc.fixed) or not same(start.slip, bc.slip):
            start = replace(bc, values=np.zeros(len(bc.fixed)), f_dead=None, p=0.0)
        zero = np.zeros(self.n_dof)
        f0 = zero if start.f_dead is None else np.asarray(start.f_dead)
        f1 = zero if bc.f_dead is None else np.asarray(bc.f_dead)
        blend = lambda s: replace(bc, values=(1 - s) * np.asarray(start.values) + s * np.asarray(bc.values),
                                  f_dead=(1 - s) * f0 + s * f1, p=(1 - s) * float(start.p) + s * float(bc.p))
        s, h, x, total = 0.0, 0.5, np.array(u0, dtype=float), 0
        while s < 1.0:
            step = min(h, 1.0 - s)
            try:
                x_new, it = self._newton(x[:self.n_dof], states, blend(s + step), tol, max_iter)
            except RuntimeError:
                h /= 2
                self.cutbacks += 1
                if h < min_step:
                    raise RuntimeError(f"FE Newton failed down to load increment {h:.1e}")
                continue
            s, x, total, h = s + step, x_new, total + it, min(2 * h, 1.0)
        return x, total

    def _newton(self, u0, states, bc, tol, max_iter):
        cons = self._constraints(bc)
        x = cons.project(np.concatenate([np.array(u0, dtype=float), self.last_p]))
        f_dead = self._f_dead(bc)
        scale = None
        for it in range(1, max_iter + 1):
            r = np.asarray(self.residual(jnp.asarray(x), states, bc.p, f_dead))
            if not np.isfinite(r).all():
                raise RuntimeError(f"FE Newton: non-finite residual in iteration {it}")
            if scale is None:
                s_u = max(np.abs(r[:self.n_dof]).max(), np.abs(np.asarray(f_dead)).max(), 1e-12)
                scale = cons.restrict_scale(np.concatenate([np.full(self.n_dof, s_u), self._p_scale]))
            r_red = cons.restrict(r)
            err = np.abs(r_red / scale).max()
            if err < tol:
                it -= 1
                break
            if err > 1e8:
                raise RuntimeError(f"FE Newton diverged, |r| = {err:.1e}")
            dx = cons.prolong(self.linear(self._K_red(x, states, bc.p, cons), r_red))
            x -= dx
            if np.abs(dx[:self.n_dof]).max() <= 1e-13 * self.length:
                break
        else:
            raise RuntimeError(f"FE Newton did not converge, |r| = {err:.2e}")
        self.last_p = x[self.n_dof:].copy()
        return x, it

    def solve(self, u0, states, bc, tol=1e-10, max_iter=30):
        """Newton -> (u, n_iter); hybrid element pressures in last_p"""
        x, it = self._solve(u0, states, bc, tol, max_iter)
        return x[:self.n_dof], it

    def equilibrium(self, states, bc, u0, tol=1e-10):
        """u with r(x, states, p) = 0 on the free unknowns, reverse-differentiable in (states, bc.p)

        adjoint:  (Z^T K Z)^T y = Z^T (u_bar, 0),  lam = Z y,   states_bar = -lam . dr/dstates,   p_bar = -lam . dr/dp
        """
        f_dead = self._f_dead(bc)

        def primal(states, p, u0):
            x, self.last_iterations = self._solve(u0, states, replace(bc, p=float(p)), tol)
            self.last_u = x[:self.n_dof]
            return jnp.asarray(x)

        @jax.custom_vjp
        def eq(states, p, u0):
            return primal(states, p, u0)[:self.n_dof]

        def fwd(states, p, u0):
            x = primal(states, p, u0)
            return x[:self.n_dof], (x, states, p)

        def bwd(res, u_bar):
            x, states, p = res
            cons = self._constraints(bc)
            rhs = np.zeros(self.n_x)
            rhs[:self.n_dof] = np.asarray(u_bar)
            lam = cons.prolong(self.linear(self._K_red(x, states, p, cons), cons.restrict(rhs), trans="T"))
            s_bar, p_bar = self.residual_vjp(x, states, p, f_dead, jnp.asarray(-lam))
            return s_bar, p_bar, jnp.zeros(self.n_dof)

        eq.defvjp(fwd, bwd)
        return eq(states, jnp.asarray(bc.p, dtype=float), jnp.asarray(u0, dtype=float))

    def run(self, states, bcs, u0=None, on_step=None, tol=1e-10, checkpoint=None):
        """Equilibrium + commit per step for a list of BCs -> (u, states, history); reverse-differentiable

        checkpoint=k: reverse mode keeps the state every k steps only and recomputes each block
        (on_step must then return JAX values and close over no traced values)
        """
        u = jnp.zeros(self.n_dof) if u0 is None else jnp.asarray(u0)
        history = []
        k = checkpoint or len(bcs)
        for s in range(0, len(bcs), max(k, 1)):
            block = self._block(bcs[s:s + k], on_step, tol)
            ps = [jnp.asarray(bc.p, dtype=float) for bc in bcs[s:s + k]]
            states, u, h = block(states, u, ps) if checkpoint else block.__wrapped__(states, u, ps)
            history += h
        return u, states, history

    def _block(self, bcs, on_step, tol):
        """Steps over bcs as one function of (states, u, pressures) whose reverse pass recomputes it"""
        def steps(states, u, ps):
            hist = []
            for bc, p in zip(bcs, ps):
                u = self.equilibrium(states, replace(bc, p=p), u, tol)
                hist.append(on_step(u, states) if on_step else u)
                states = self.commit(u, states)
            return states, u, hist

        block = jax.custom_vjp(steps)
        block.defvjp(lambda *args: (steps(*args), args), lambda res, ct: jax.vjp(steps, *res)[1](ct))
        block.__wrapped__ = steps
        return block
