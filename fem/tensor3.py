"""3x3 tensor helpers (batched over leading axes): closed-form determinant and inverse, polar rotation."""

import jax
import jax.numpy as jnp


def det3(A):
    """det A = A0 . (A1 x A2) over the rows"""
    return jnp.sum(A[..., 0, :] * jnp.cross(A[..., 1, :], A[..., 2, :]), axis=-1)


def inv3(A):
    """A^-1 = [A1 x A2, A2 x A0, A0 x A1] / det A  (columns)"""
    c = jnp.stack([jnp.cross(A[..., 1, :], A[..., 2, :]), jnp.cross(A[..., 2, :], A[..., 0, :]),
                   jnp.cross(A[..., 0, :], A[..., 1, :])], axis=-1)
    return c / det3(A)[..., None, None]


@jax.jit
def polar_rotation(F, n_iter=12):
    """F = R U,  R_{k+1} = (R_k + R_k^-T)/2 from R_0 = F, smooth at repeated singular values"""
    return jax.lax.fori_loop(0, n_iter, lambda _, R: 0.5 * (R + inv3(R).T), F)
