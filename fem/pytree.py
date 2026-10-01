"""Pytree registration for plain classes holding model state"""

import jax


def pytree(data, static=()):
    """Register a class as pytree: data attributes are leaves, static ones aux data; adds replace(**fields)"""
    def wrap(cls):
        def flatten(obj):
            return tuple(getattr(obj, k) for k in data), tuple(getattr(obj, k) for k in static)

        def unflatten(aux, children):
            obj = cls.__new__(cls)
            for k, v in zip(data, children):
                setattr(obj, k, v)
            for k, v in zip(static, aux):
                setattr(obj, k, v)
            return obj

        def replace(self, **fields):
            children, aux = flatten(self)
            obj = unflatten(aux, children)
            for k, v in fields.items():
                setattr(obj, k, v)
            return obj

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        cls.replace = replace
        return cls
    return wrap
