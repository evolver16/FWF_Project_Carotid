"""Total-Lagrangian hex8 FE: kernels, assembly, follower pressure, Newton, G&R time loop.

Model interface per integration point (hcmm, fcmm, elastic):
    sigma, aux = model.sigma_solver(state, F)
    state      = model.commit(state, F, aux)
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import setups  # noqa: F401  (x64, compilation cache)
from mesh import HEX_CORNERS
from tensor3 import det3, inv3

GP3 = HEX_CORNERS / np.sqrt(3.0)
QUAD = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
GP2 = QUAD / np.sqrt(3.0)


def hex_dN(xi):
    """dN_a/dxi_j of the trilinear hex8, (8, 3)"""
    c = HEX_CORNERS
    t = 1 + c * xi
    return np.stack([c[:, 0] * t[:, 1] * t[:, 2], t[:, 0] * c[:, 1] * t[:, 2],
                     t[:, 0] * t[:, 1] * c[:, 2]], axis=1) / 8


def quad_N(eta):
    """N_a and dN_a/deta_j of the bilinear quad, (4,), (4, 2)"""
    t = 1 + QUAD * eta
    return t[:, 0] * t[:, 1] / 4, np.stack([QUAD[:, 0] * t[:, 1], t[:, 0] * QUAD[:, 1]], axis=1) / 4


def geometry(X, conn):
    """dN/dX and w det(dX/dxi) at the 2x2x2 Gauss points, (n_el, 8, 8, 3), (n_el, 8)"""
    dNdxi = np.stack([hex_dN(g) for g in GP3])
    J0 = np.einsum("eai,gaj->egij", X[conn], dNdxi)
    return np.einsum("gaj,egji->egai", dNdxi, np.linalg.inv(J0)), np.linalg.det(J0)


def gauss_points(X, conn):
    """Reference coordinates of the 2x2x2 Gauss points, (n_el, 8, 3)"""
    N = np.stack([np.prod(1 + HEX_CORNERS * g, axis=1) / 8 for g in GP3])
    return np.einsum("ga,eai->egi", N, X[conn])


def broadcast_state(state, n_el):
    """One material state copied to every integration point, leaves (n_el, 8, ...)"""
    return jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x), (n_el, 8) + jnp.shape(x)), state)


@dataclass
class BC:
    """Dirichlet dofs/values, dead nodal forces, follower pressure p on the pressure faces."""
    fixed: np.ndarray
    values: np.ndarray
    f_dead: np.ndarray = None
    p: float = 0.0


class System:
    def __init__(self, mesh, model, pressure_faces=None, batch=128):
        self.mesh, self.model, self.batch = mesh, model, batch
        self.n_dof = 3 * mesh.n_nodes
        dNdX, wdet = geometry(mesh.X, mesh.conn)
        self.dNdX, self.wdet = jnp.asarray(dNdX), jnp.asarray(wdet)
        self.conn = jnp.asarray(mesh.conn)
        self.faces = jnp.asarray(pressure_faces if pressure_faces is not None
                                 else np.zeros((0, 4), dtype=int))
        self.X = jnp.asarray(mesh.X)
        self.length = float(np.ptp(mesh.X, axis=0).max())
        N2, dN2 = zip(*[quad_N(g) for g in GP2])
        self.N2, self.dN2 = jnp.asarray(np.stack(N2)), jnp.asarray(np.stack(dN2))

        edofs = (3 * mesh.conn[:, :, None] + np.arange(3)).reshape(mesh.n_elem, 24)
        fdofs = (3 * np.asarray(self.faces)[:, :, None] + np.arange(3)).reshape(-1, 12)
        self._rows = np.concatenate([np.repeat(edofs, 24, axis=1).ravel(),
                                     np.repeat(fdofs, 12, axis=1).ravel()])
        self._cols = np.concatenate([np.tile(edofs, 24).ravel(), np.tile(fdofs, 12).ravel()])

        self.residual = jax.jit(self._residual)
        self.tangent = jax.jit(self._tangent)
        self.gauss_F = jax.jit(self._gauss_F)
        self.stress = jax.jit(self._stress)
        self.commit = jax.jit(jax.checkpoint(self._commit))
        self.residual_vjp = jax.jit(self._residual_vjp)
        self.last_u, self.last_iterations = None, 0

    def _gauss_F(self, u):
        """F = I + grad_X u at every Gauss point, (n_el, 8, 3, 3)"""
        return jnp.eye(3) + jnp.einsum("eai,egaj->egij", u.reshape(-1, 3)[self.conn], self.dNdX)

    def _elem_force(self, u_e, st_e, dNdX_e, w_e):
        """f_ai = sum_g w_g P_iJ dN_a/dX_J,  P = J sigma F^-T"""
        F = jnp.eye(3) + jnp.einsum("ai,gaj->gij", u_e, dNdX_e)
        sig = jax.vmap(lambda s, f: self.model.sigma_solver(s, f)[0])(st_e, F)
        P = det3(F)[:, None, None] * sig @ inv3(F).transpose(0, 2, 1)
        return jnp.einsum("g,gaJ,giJ->ai", w_e, dNdX_e, P)

    def _face_force(self, x_f, p):
        """f_a = -p sum_g N_a (dx/deta1 x dx/deta2)   (follower pressure on the deformed face)"""
        a1 = jnp.einsum("ai,ga->gi", x_f, self.dN2[:, :, 0])
        a2 = jnp.einsum("ai,ga->gi", x_f, self.dN2[:, :, 1])
        return -p * jnp.einsum("ga,gi->ai", self.N2, jnp.cross(a1, a2))

    def _residual(self, u, states, p, f_dead):
        """r = f_int(u) - f_pressure(u) - f_dead"""
        U = u.reshape(-1, 3)
        fe = jax.lax.map(lambda a: self._elem_force(*a), (U[self.conn], states, self.dNdX, self.wdet),
                         batch_size=self.batch)
        r = jnp.zeros_like(U).at[self.conn].add(fe)
        ff = jax.vmap(self._face_force, (0, None))((self.X + U)[self.faces], p)
        r = r.at[self.faces].add(-ff)
        return r.ravel() - f_dead

    def _residual_vjp(self, u, states, p, f_dead, w):
        """(w . dr/dstates, w . dr/dp)"""
        return jax.vjp(lambda s, q: self._residual(u, s, q, f_dead), states, p)[1](w)

    def _P(self, st, F):
        """P = J sigma F^-T"""
        return det3(F) * self.model.sigma_solver(st, F)[0] @ inv3(F).T

    def _elem_tangent(self, u_e, st_e, dNdX_e, w_e):
        """K_aibk = sum_g w_g dN_a/dX_J (dP_iJ/dF_kL) dN_b/dX_L,  dP/dF by forward AD per Gauss point"""
        F = jnp.eye(3) + jnp.einsum("ai,gaj->gij", u_e, dNdX_e)
        A = jax.vmap(jax.jacfwd(self._P, argnums=1))(st_e, F)
        return jnp.einsum("g,gaJ,giJkL,gbL->aibk", w_e, dNdX_e, A, dNdX_e).reshape(24, 24)

    def _tangent(self, u, states, p):
        """Element and face tangents dr/du for sparse assembly"""
        U = u.reshape(-1, 3)
        ke = jax.lax.map(lambda a: self._elem_tangent(*a), (U[self.conn], states, self.dNdX, self.wdet),
                         batch_size=self.batch)
        kf = jax.vmap(jax.jacfwd(lambda xf: -self._face_force(xf.reshape(4, 3), p).ravel()))(
            (self.X + U)[self.faces].reshape(-1, 12))
        return jnp.concatenate([ke.ravel(), kf.ravel()])

    def _stress(self, u, states):
        """Cauchy stress at every Gauss point, (n_el, 8, 3, 3)"""
        F = self._gauss_F(u)
        return jax.vmap(jax.vmap(lambda s, f: self.model.sigma_solver(s, f)[0]))(states, F)

    def _commit(self, u, states):
        F = self._gauss_F(u)
        aux = jax.vmap(jax.vmap(lambda s, f: self.model.sigma_solver(s, f)[1]))(states, F)
        return jax.vmap(jax.vmap(self.model.commit))(states, F, aux)

    def _free(self, bc):
        free = np.ones(self.n_dof, dtype=bool)
        free[bc.fixed] = False
        return free

    def _f_dead(self, bc):
        return jnp.zeros(self.n_dof) if bc.f_dead is None else jnp.asarray(bc.f_dead)

    def _K_free(self, u, states, p, free):
        vals = np.asarray(self.tangent(jnp.asarray(u), states, p))
        K = sp.csr_matrix((vals, (self._rows, self._cols)), shape=(self.n_dof, self.n_dof))
        return K[free][:, free]

    def solve(self, u0, states, bc, tol=1e-10, max_iter=30):
        """Newton on the free dofs -> (u, n_iter)"""
        u = np.array(u0, dtype=float)
        u[bc.fixed] = bc.values
        free = self._free(bc)
        f_dead = self._f_dead(bc)
        scale = None
        for it in range(1, max_iter + 1):
            r = np.asarray(self.residual(jnp.asarray(u), states, bc.p, f_dead))
            if scale is None:
                scale = max(np.abs(r).max(), np.abs(np.asarray(f_dead)).max(), 1e-12)
            if np.abs(r[free]).max() < tol * scale:
                return u, it - 1
            du = spla.spsolve(self._K_free(u, states, bc.p, free).tocsc(), r[free])
            u[free] -= du
            if np.abs(du).max() <= 1e-13 * self.length:
                return u, it
        raise RuntimeError(f"FE Newton did not converge, |r| = {np.abs(r[free]).max():.2e}")

    def equilibrium(self, states, bc, u0, tol=1e-10):
        """u with r(u, states, p) = 0 on the free dofs, reverse-differentiable in (states, bc.p)

        adjoint:  K_ff^T lam_f = u_bar_f,   states_bar = -lam . dr/dstates,   p_bar = -lam . dr/dp
        """
        free, f_dead = self._free(bc), self._f_dead(bc)

        @jax.custom_vjp
        def eq(states, p, u0):
            self.last_u, self.last_iterations = self.solve(u0, states, replace(bc, p=float(p)), tol)
            return jnp.asarray(self.last_u)

        def fwd(states, p, u0):
            u = eq(states, p, u0)
            return u, (u, states, p)

        def bwd(res, u_bar):
            u, states, p = res
            lam = np.zeros(self.n_dof)
            lam[free] = spla.spsolve(self._K_free(u, states, p, free).T.tocsc(), np.asarray(u_bar)[free])
            s_bar, p_bar = self.residual_vjp(u, states, p, f_dead, jnp.asarray(-lam))
            return s_bar, p_bar, jnp.zeros(self.n_dof)

        eq.defvjp(fwd, bwd)
        return eq(states, jnp.asarray(bc.p, dtype=float), jnp.asarray(u0, dtype=float))

    def run(self, states, bcs, u0=None, on_step=None, tol=1e-10):
        """Equilibrium + commit per step for a list of BCs -> (u, states, history); reverse-differentiable"""
        u = jnp.zeros(self.n_dof) if u0 is None else jnp.asarray(u0)
        history = []
        for bc in bcs:
            u = self.equilibrium(states, bc, u, tol)
            history.append(on_step(u, states) if on_step else u)
            states = self.commit(u, states)
        return u, states, history
