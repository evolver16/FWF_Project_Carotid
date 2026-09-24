"""Structured hex8 meshes, node sets and outward-oriented boundary faces."""

from dataclasses import dataclass, field

import numpy as np

# hex8 node order: bottom (xi2 = -1) counter-clockwise, then top
HEX_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
HEX_FACES = {(0, -1): [0, 3, 7, 4], (0, 1): [1, 2, 6, 5],
             (1, -1): [0, 1, 5, 4], (1, 1): [3, 7, 6, 2],
             (2, -1): [0, 1, 2, 3], (2, 1): [4, 5, 6, 7]}


@dataclass
class Mesh:
    """nodes/faces keyed (index axis, side) with side -1 / +1; faces are outward-oriented quads."""
    X: np.ndarray
    conn: np.ndarray
    nodes: dict = field(default_factory=dict)
    faces: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return self.X.shape[0]

    @property
    def n_elem(self):
        return self.conn.shape[0]


def structured(n, mapping):
    """Hex8 grid on the unit cube in index space, node coordinates X = mapping(s), s in [0,1]^3."""
    ijk = np.stack(np.meshgrid(*[np.arange(n[a] + 1) for a in range(3)], indexing="ij"),
                   axis=-1).reshape(-1, 3)
    X = mapping(ijk / np.array(n, dtype=float))
    idx = lambda t: (t[0] * (n[1] + 1) + t[1]) * (n[2] + 1) + t[2]
    conn = np.array([[idx((i + (a > 0), j + (b > 0), k + (c > 0))) for a, b, c in HEX_CORNERS]
                     for i in range(n[0]) for j in range(n[1]) for k in range(n[2])])

    nodes = {(a, s): np.where(ijk[:, a] == (0 if s < 0 else n[a]))[0] for a in range(3) for s in (-1, 1)}
    centroids = X[conn].mean(axis=1)
    faces = {}
    for (a, s), loc in HEX_FACES.items():
        on = np.all(np.isin(conn[:, loc], nodes[(a, s)]), axis=1)
        quads = conn[on][:, loc]
        x = X[quads]
        normal = np.cross(x[:, 1] - x[:, 0], x[:, 3] - x[:, 0])
        flip = np.einsum("ij,ij->i", normal, x.mean(axis=1) - centroids[on]) < 0
        quads[flip] = quads[flip][:, ::-1]
        faces[(a, s)] = quads
    return Mesh(X=X, conn=conn, nodes=nodes, faces=faces)


def box(n=(1, 1, 1), L=(50.0, 50.0, 50.0), perturb=0.0, seed=0):
    """Box [0,L0]x[0,L1]x[0,L2]; interior nodes optionally perturbed by perturb * h."""
    m = structured(n, lambda s: s * np.array(L))
    if perturb > 0:
        ijk = np.round(m.X / (np.array(L) / np.array(n))).astype(int)
        interior = np.all((ijk > 0) & (ijk < np.array(n)), axis=1)
        h = np.array(L) / np.array(n)
        m.X[interior] += perturb * h * np.random.default_rng(seed).uniform(-1, 1, (interior.sum(), 3))
    return m


def quarter_cylinder(r_i=5.0, t=1.3, L=0.22, n=(8, 60, 4)):
    """Index axes (r, theta, z); theta in [0, pi/2], cylinder axis = z.

    nodes/faces: (0,-1) inner, (0,1) outer, (1,-1) theta = 0 (y = 0), (1,1) theta = pi/2 (x = 0), (2,+-1) ends.
    """
    def mapping(s):
        r, th, z = r_i + t * s[:, 0], 0.5 * np.pi * s[:, 1], L * s[:, 2]
        return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    return structured(n, mapping)


def cylinder_basis(x):
    """Columns (e_r, e_theta, e_z) at points x, (..., 3, 3)"""
    th = np.arctan2(x[..., 1], x[..., 0])
    c, s, o, i = np.cos(th), np.sin(th), np.zeros_like(th), np.ones_like(th)
    return np.stack([np.stack([c, s, o], -1), np.stack([-s, c, o], -1), np.stack([o, o, i], -1)], -1)
