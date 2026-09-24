"""Closed-form 3x3 determinant and inverse (batched over leading axes)."""

import jax.numpy as jnp


def det3(A):
    """det A = A0 . (A1 x A2) over the rows"""
    return jnp.sum(A[..., 0, :] * jnp.cross(A[..., 1, :], A[..., 2, :]), axis=-1)


def inv3(A):
    """A^-1 = [A1 x A2, A2 x A0, A0 x A1] / det A  (columns)"""
    c = jnp.stack([jnp.cross(A[..., 1, :], A[..., 2, :]), jnp.cross(A[..., 2, :], A[..., 0, :]),
                   jnp.cross(A[..., 0, :], A[..., 1, :])], axis=-1)
    return c / det3(A)[..., None, None]
