"""Centerline and tortuosity index of a tube mesh (Weiss et al. 2020)."""

import numpy as np


def centerline(m, u=None, sym=None):
    """Centroids of the deformed rings x = X + u along the axial index (Mesh.param[:, 2]), (n_stations, 3);
    sym: zero the y-coordinate (half tube, symmetry plane y = 0), default: m has a "sym" node set"""
    x = m.X if u is None else m.X + np.asarray(u).reshape(-1, 3)
    s, ring = np.unique(np.round(m.param[:, 2], 12), return_inverse=True)
    c = np.stack([np.bincount(ring, x[:, i]) for i in range(3)], -1) / np.bincount(ring)[:, None]
    if ("sym" in m.nodes) if sym is None else sym:
        c[:, 1] = 0.0
    return c


def tortuosity(c):
    """TI = (L1 - L2) / L2,  L1 = arc length of the centerline c (n, 3), L2 = |c_n - c_1|"""
    L1 = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
    L2 = np.linalg.norm(c[-1] - c[0])
    return (L1 - L2) / L2
