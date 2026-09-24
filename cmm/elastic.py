"""Compressible neo-Hookean without G&R, same interface as hcmm/fcmm (sigma_solver, commit)."""

import jax
import jax.numpy as jnp
from jax import jit
from tensor3 import det3


@jax.tree_util.register_pytree_node_class
class NeoHookean:
    def __init__(self, C10, K):
        self.C10 = jnp.asarray(C10)
        self.K = jnp.asarray(K)

    def tree_flatten(self):
        return (self.C10, self.K), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def Psi(self, F):
        """W = C10 (J^(-2/3) tr(F^T F) - 3) + K/2 (J - 1)^2"""
        J = det3(F)
        return self.C10 * (J ** (-2 / 3) * jnp.trace(F.T @ F) - 3) + self.K / 2 * (J - 1) ** 2


@jax.tree_util.register_pytree_node_class
class NeoHookeanInc:
    def __init__(self, C10):
        self.C10 = jnp.asarray(C10)

    def tree_flatten(self):
        return (self.C10,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def Psi(self, F):
        """W = C10 (tr(F^T F) - 3), pressure from the hybrid element"""
        return self.C10 * (jnp.trace(F.T @ F) - 3)


def J_target(state):
    return jnp.ones(())


@jit
def sigma_solver(state, F):
    """sigma = (1/J) (dW/dF) F^T"""
    return jax.grad(state.Psi)(F) @ F.T / det3(F), None


@jit
def commit(state, F, aux):
    return state
