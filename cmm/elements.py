"""Element and face types: shape functions, Gauss rules, Abaqus face numbering, VTK cell types."""

from dataclasses import dataclass

import numpy as np

HEX_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
QUAD_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


@dataclass(frozen=True)
class Face:
    name: str
    n_nodes: int
    n_corners: int
    gauss: np.ndarray
    weights: np.ndarray
    flip: tuple

    def N(self, eta):
        eta = np.asarray(eta, dtype=float)
        if self.name == "quad4":
            t = 1 + QUAD_CORNERS * eta
            return t[:, 0] * t[:, 1] / 4
        L = np.array([1 - eta[0] - eta[1], eta[0], eta[1]])
        return np.concatenate([L * (2 * L - 1), 4 * L * np.roll(L, -1)])

    def dN(self, eta):
        """dN_a/deta_j, (n, 2)"""
        eta = np.asarray(eta, dtype=float)
        if self.name == "quad4":
            t = 1 + QUAD_CORNERS * eta
            return np.stack([QUAD_CORNERS[:, 0] * t[:, 1], t[:, 0] * QUAD_CORNERS[:, 1]], axis=1) / 4
        L = np.array([1 - eta[0] - eta[1], eta[0], eta[1]])
        dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        corner = (4 * L - 1)[:, None] * dL
        Ln, dLn = np.roll(L, -1), np.roll(dL, -1, axis=0)
        mid = 4 * (dL * Ln[:, None] + L[:, None] * dLn)
        return np.concatenate([corner, mid])


@dataclass(frozen=True)
class Element:
    name: str
    n_nodes: int
    gauss: np.ndarray
    weights: np.ndarray
    centre: np.ndarray
    faces: dict
    face: Face
    vtk: int
    abaqus: tuple

    def N(self, xi):
        xi = np.asarray(xi, dtype=float)
        if self.name == "hex8":
            return np.prod(1 + HEX_CORNERS * xi, axis=1) / 8
        L = np.array([1 - xi.sum(), *xi])
        return np.concatenate([L * (2 * L - 1), [4 * L[i] * L[j] for i, j in TET10_EDGES]])

    def dN(self, xi):
        """dN_a/dxi_j, (n, 3)"""
        xi = np.asarray(xi, dtype=float)
        if self.name == "hex8":
            c, t = HEX_CORNERS, 1 + HEX_CORNERS * xi
            return np.stack([c[:, 0] * t[:, 1] * t[:, 2], t[:, 0] * c[:, 1] * t[:, 2],
                             t[:, 0] * t[:, 1] * c[:, 2]], axis=1) / 8
        L = np.array([1 - xi.sum(), *xi])
        dL = np.vstack([-np.ones(3), np.eye(3)])
        corner = (4 * L - 1)[:, None] * dL
        mid = np.array([4 * (dL[i] * L[j] + L[i] * dL[j]) for i, j in TET10_EDGES])
        return np.concatenate([corner, mid])


QUAD4 = Face("quad4", 4, 4, QUAD_CORNERS / np.sqrt(3.0), np.ones(4), (3, 2, 1, 0))
_a, _b = 0.445948490915965, 0.091576213509771
TRI6 = Face("tri6", 6, 3,
            np.array([[_a, _a], [1 - 2 * _a, _a], [_a, 1 - 2 * _a], [_b, _b], [1 - 2 * _b, _b], [_b, 1 - 2 * _b]]),
            np.array([0.223381589678011] * 3 + [0.109951743655322] * 3) / 2, (0, 2, 1, 5, 4, 3))

_ta, _tb = 0.5854101966249685, 0.1381966011250105
HEX8 = Element("hex8", 8, HEX_CORNERS / np.sqrt(3.0), np.ones(8), np.zeros(3),
               {1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [0, 1, 5, 4], 4: [1, 2, 6, 5], 5: [2, 3, 7, 6], 6: [3, 0, 4, 7]},
               QUAD4, 12, ("C3D8", "C3D8H", "C3D8R", "C3D8RH", "C3D8I", "C3D8IH"))
TET10 = Element("tet10", 10,
                np.array([[_tb, _tb, _tb], [_ta, _tb, _tb], [_tb, _ta, _tb], [_tb, _tb, _ta]]),
                np.full(4, 1 / 24), np.full(3, 0.25),
                {1: [0, 1, 2, 4, 5, 6], 2: [0, 3, 1, 7, 8, 4], 3: [1, 3, 2, 8, 9, 5], 4: [2, 3, 0, 9, 7, 6]},
                TRI6, 24, ("C3D10", "C3D10H", "C3D10M", "C3D10MH"))
ELEMENT_TYPES = {"hex8": HEX8, "tet10": TET10}


def element_for(n_nodes):
    return {8: HEX8, 10: TET10}[n_nodes]


def face_for(n_nodes):
    return {4: QUAD4, 6: TRI6}[n_nodes]
