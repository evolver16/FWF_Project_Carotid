"""Symmetric (3,3) <-> Voigt (11,22,33,12,13,23)"""

import jax.numpy as jnp


def sym_to_voigt(T):
    """(3,3) -> (11,22,33,12,13,23)"""
    return jnp.array([T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2]])

def voigt_to_sym(v):
    """(11,22,33,12,13,23) -> (3,3)"""
    return jnp.array([[v[0], v[3], v[4]],
                       [v[3], v[1], v[5]],
                       [v[4], v[5], v[2]]])
