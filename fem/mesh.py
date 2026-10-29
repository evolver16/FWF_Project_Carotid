"""Meshes of hex8, tet10 or tet4: structured generators, boundary faces, Abaqus .inp and gmsh .msh I/O, VTK .vtu/.pvd."""

import base64
import itertools
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from fem.elements import HEX_CORNERS, ELEMENT_TYPES, element_for, face_for

HEX_FACES = {(0, -1): [0, 3, 7, 4], (0, 1): [1, 2, 6, 5],
             (1, -1): [0, 1, 5, 4], (1, 1): [3, 7, 6, 2],
             (2, -1): [0, 1, 2, 3], (2, 1): [4, 5, 6, 7]}
TET10_SWAP = [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]
GMSH_TYPES = {5: 8, 11: 10, 3: 4, 9: 6}
GMSH_ORDER = {10: [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]}


@dataclass
class Mesh:
    """nodes/faces/elsets keyed by name or (index axis, side); faces are outward-oriented quad4 or tri6."""
    X: np.ndarray
    conn: np.ndarray
    nodes: dict = field(default_factory=dict)
    faces: dict = field(default_factory=dict)
    elsets: dict = field(default_factory=dict)
    param: np.ndarray = None

    @property
    def n_nodes(self):
        return self.X.shape[0]

    @property
    def n_elem(self):
        return self.conn.shape[0]

    @property
    def element(self):
        return element_for(self.conn.shape[1])


def structured(n, mapping, etype="hex8"):
    """Grid on the unit cube in index space, X = mapping(s), s in [0,1]^3 (kept as Mesh.param);
    tet10 / tet4: 6 Kuhn tets per cell, midside nodes mapped from index space (curved edges follow the mapping)."""
    n = np.asarray(n)
    if etype == "hex8":
        ijk = np.stack(np.meshgrid(*[np.arange(n[a] + 1) for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
        idx = lambda t: (t[0] * (n[1] + 1) + t[1]) * (n[2] + 1) + t[2]
        conn = np.array([[idx((i + (a > 0), j + (b > 0), k + (c > 0))) for a, b, c in HEX_CORNERS]
                         for i in range(n[0]) for j in range(n[1]) for k in range(n[2])])
        grid, top = ijk, n
    else:
        cells = 2 * np.stack(np.meshgrid(*[np.arange(n[a]) for a in range(3)], indexing="ij"), -1).reshape(-1, 1, 3)
        paths = []
        for p in itertools.permutations(range(3)):
            v = [np.zeros(3, dtype=int)]
            for axis in p:
                v.append(v[-1] + 2 * np.eye(3, dtype=int)[axis])
            paths.append(v)
        corners = (cells[:, None] + np.array(paths)[None]).reshape(-1, 4, 3)
        mids = np.stack([(corners[:, i] + corners[:, j]) // 2 for i, j in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]], 1)
        nodes_e = np.concatenate([corners, mids], 1) if etype == "tet10" else corners
        grid, inv = np.unique(nodes_e.reshape(-1, 3), axis=0, return_inverse=True)
        conn = inv.reshape(len(nodes_e), -1)
        top = 2 * n
    X = mapping(grid / top.astype(float))
    if etype != "hex8":
        x = X[conn[:, :4]]
        neg = np.linalg.det(np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], 1)) < 0
        conn[neg] = conn[neg][:, TET10_SWAP[:conn.shape[1]]]

    nodes = {(a, s): np.flatnonzero(grid[:, a] == (0 if s < 0 else top[a])) for a in range(3) for s in (-1, 1)}
    el = ELEMENT_TYPES[etype]
    faces = {}
    for key, set_ in nodes.items():
        if etype == "hex8":
            loc = HEX_FACES[key]
            on = np.flatnonzero(np.all(np.isin(conn[:, loc], set_), axis=1))
            faces[key] = outward(X, conn, on, conn[on][:, loc])
        else:
            found = [(e, conn[e][loc]) for loc in el.faces.values()
                     for e in np.flatnonzero(np.all(np.isin(conn[:, loc], set_), axis=1))]
            elems = np.array([e for e, _ in found])
            faces[key] = outward(X, conn, elems, np.array([f for _, f in found]))
    return Mesh(X=X, conn=conn, nodes=nodes, faces=faces, param=grid / top.astype(float))


def outward(X, conn, elems, faces):
    """Faces reordered so that (x1 - x0) x (x_last corner - x0) points away from their element centroids"""
    face = face_for(faces.shape[1])
    x = X[faces]
    normal = np.cross(x[:, 1] - x[:, 0], x[:, face.n_corners - 1] - x[:, 0])
    flip = np.einsum("ij,ij->i", normal, x.mean(axis=1) - X[conn[elems]].mean(axis=1)) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, list(face.flip)]
    return faces


def boundary_faces(m):
    """(element, Abaqus face number, outward face) of every face that belongs to one element only"""
    el = m.element
    loc = np.array(list(el.faces.values()))
    faces = m.conn[:, loc].reshape(-1, loc.shape[1])
    _, inv, count = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    b = np.flatnonzero(count[inv.ravel()] == 1)
    elems, face = b // len(loc), b % len(loc) + 1
    return elems, face, outward(m.X, m.conn, elems, faces[b])


def _ids(tokens):
    return [int(t) for t in tokens if t.strip()]


def read_inp(path):
    """Abaqus .inp (single part, C3D8*, C3D10* or C3D4*) -> Mesh with *Nset, *Elset and element-based *Surface by name;
    nodes in assembly coordinates (*Instance translation, then rotation about the axis a -> b by the angle in degrees)."""
    node_id, X, elem_id, conn, instance = [], [], [], [], []
    nsets, elsets, surfaces = {}, {}, {}
    block, opts, pending, nen = None, {}, [], None
    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            parts = [p.strip() for p in line[1:].split(",")]
            block = parts[0].lower()
            opts = {p.split("=")[0].strip().lower(): (p.split("=")[1].strip() if "=" in p else True)
                    for p in parts[1:]}
            if block == "element":
                etype = str(opts.get("type", "")).upper()
                el = next((e for e in ELEMENT_TYPES.values() if etype in e.abaqus), None)
                if el is None or (nen is not None and el.n_nodes != nen):
                    raise ValueError(f"element type {etype} not supported (one of C3D8*, C3D10*, C3D4*)")
                nen = el.n_nodes
            if block == "surface":
                surfaces[opts["name"]] = []
            continue
        tok = [t.strip() for t in line.rstrip(",").split(",")]
        if block == "instance":
            instance.append([float(t) for t in tok])
            continue
        if block == "node":
            node_id.append(int(tok[0]))
            X.append([float(t) for t in tok[1:4]])
        elif block == "element":
            pending += _ids(tok)
            if len(pending) >= nen + 1:
                elem_id.append(pending[0])
                conn.append(pending[1:nen + 1])
                if "elset" in opts:
                    elsets.setdefault(opts["elset"], []).append(pending[0])
                pending = []
        elif block in ("nset", "elset"):
            sets, name = (nsets, opts["nset"]) if block == "nset" else (elsets, opts["elset"])
            if opts.get("generate"):
                a, b, *s = _ids(tok)
                ids = list(range(a, b + 1, s[0] if s else 1))
            else:
                ids = [i for t in tok for i in (sets.get(t, []) if not t.lstrip("-").isdigit() else [int(t)])]
            sets.setdefault(name, []).extend(ids)
        elif block == "surface" and len(tok) > 1:
            surfaces[opts["name"]].append((tok[0], int(tok[1].upper().lstrip("S"))))

    nmap = {i: k for k, i in enumerate(node_id)}
    emap = {i: k for k, i in enumerate(elem_id)}
    X = np.array(X)
    if instance:
        X = X + np.array(instance[0][:3])
    if len(instance) > 1:
        a, b, ang = np.array(instance[1][:3]), np.array(instance[1][3:6]), np.radians(instance[1][6])
        k = (b - a) / np.linalg.norm(b - a)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        X = a + (X - a) @ R.T
    conn = np.vectorize(nmap.get)(np.array(conn))
    m = Mesh(X=X, conn=conn,
             nodes={k: np.array(sorted({nmap[i] for i in v})) for k, v in nsets.items()},
             elsets={k: np.array(sorted({emap[i] for i in v})) for k, v in elsets.items()})
    for name, entries in surfaces.items():
        faces = []
        for ref, s in entries:
            el = m.elsets[ref] if ref in m.elsets else np.array([emap[int(ref)]])
            faces.append(outward(X, conn, el, conn[el][:, m.element.faces[s]]))
        m.faces[name] = np.concatenate(faces)
    return m


def write_inp(path, m):
    """Abaqus .inp with the string-named node sets and faces of m (faces as element-based surfaces)."""
    elems, face, faces = boundary_faces(m)
    by_nodes = {tuple(sorted(q)): (e, s) for e, s, q in zip(elems, face, faces)}
    out = ["*Heading", "cmm mesh", "*Node"]
    out += [f"{i + 1}, {x:.12g}, {y:.12g}, {z:.12g}" for i, (x, y, z) in enumerate(m.X)]
    out.append(f"*Element, type={m.element.abaqus[0]}, elset=ALL")
    out += [f"{e + 1}, " + ", ".join(str(n + 1) for n in c) for e, c in enumerate(m.conn)]

    def ids(v):
        v = np.asarray(v) + 1
        return [", ".join(map(str, v[k:k + 16])) for k in range(0, len(v), 16)]

    for name, v in m.nodes.items():
        if isinstance(name, str):
            out += [f"*Nset, nset={name}"] + ids(v)
    for name, v in m.elsets.items():
        if isinstance(name, str):
            out += [f"*Elset, elset={name}"] + ids(v)
    surfaces = []
    for name, q in m.faces.items():
        if not isinstance(name, str):
            continue
        es = np.array([by_nodes[tuple(sorted(f))] for f in q])
        surfaces += [f"*Surface, type=ELEMENT, name={name}"]
        for s in np.unique(es[:, 1]):
            out += [f"*Elset, elset=_{name}_S{s}, internal"] + ids(es[es[:, 1] == s, 0])
            surfaces += [f"_{name}_S{s}, S{s}"]
    pathlib.Path(path).write_text("\n".join(out + surfaces) + "\n")


def read_msh(path):
    """gmsh .msh 2.2 or 4.1 (ASCII) -> Mesh; physical volumes -> elsets, physical surfaces -> faces and nodes,
    lower-dimensional physical groups -> nodes (hex8 or tet10 volumes, quad4 or tri6 surfaces)"""
    lines = iter(pathlib.Path(path).read_text().split("\n"))
    names, ent_phys = {}, {}
    node_tag, X, blocks = [], [], []
    version = None
    for line in lines:
        s = line.strip()
        if s == "$MeshFormat":
            version = next(lines).split()[0]
        elif s == "$PhysicalNames":
            for _ in range(int(next(lines))):
                dim, tag, name = next(lines).split(maxsplit=2)
                names[(int(dim), int(tag))] = name.strip().strip('"')
        elif s == "$Entities":
            counts = [int(c) for c in next(lines).split()]
            for dim, count in enumerate(counts):
                for _ in range(count):
                    t = next(lines).split()
                    k = 4 if dim == 0 else 7
                    ent_phys[(dim, int(t[0]))] = [int(v) for v in t[k + 1:k + 1 + int(t[k])]]
        elif s == "$Nodes":
            if version.startswith("2"):
                for _ in range(int(next(lines))):
                    t = next(lines).split()
                    node_tag.append(int(t[0]))
                    X.append([float(v) for v in t[1:4]])
            else:
                n_blocks = int(next(lines).split()[0])
                for _ in range(n_blocks):
                    count = int(next(lines).split()[3])
                    tags = [int(next(lines)) for _ in range(count)]
                    node_tag += tags
                    X += [[float(v) for v in next(lines).split()[:3]] for _ in range(count)]
        elif s == "$Elements":
            if version.startswith("2"):
                for _ in range(int(next(lines))):
                    t = [int(v) for v in next(lines).split()]
                    etype, ntags = t[1], t[2]
                    if etype in GMSH_TYPES:
                        blocks.append((etype, [t[3]] if ntags else [], t[3 + ntags:]))
            else:
                n_blocks = int(next(lines).split()[0])
                for _ in range(n_blocks):
                    dim, ent, etype, count = [int(v) for v in next(lines).split()]
                    for _ in range(count):
                        t = [int(v) for v in next(lines).split()]
                        if etype in GMSH_TYPES:
                            blocks.append((etype, ent_phys.get((dim, ent), []), t[1:]))
    nmap = {t: k for k, t in enumerate(node_tag)}
    to_idx = lambda nodes, nen: np.array([nmap[v] for v in nodes])[GMSH_ORDER.get(nen, slice(None))]
    vol = [(phys, to_idx(n, GMSH_TYPES[e])) for e, phys, n in blocks if GMSH_TYPES[e] in (8, 10)]
    if len({len(c) for _, c in vol}) != 1:
        raise ValueError("mesh must contain one volume element type (hex8 or tet10)")
    m = Mesh(X=np.array(X), conn=np.array([c for _, c in vol]))
    dim_of = lambda nen: 3 if nen in (8, 10) else 2
    for k, (phys, _) in enumerate(vol):
        for p in phys:
            m.elsets.setdefault(names.get((3, p), f"volume_{p}"), []).append(k)
    m.elsets = {k: np.array(v) for k, v in m.elsets.items()}
    _, _, bfaces = boundary_faces(m)
    by_nodes = {tuple(sorted(f)): f for f in bfaces}
    surf = {}
    for e, phys, nodes in blocks:
        nen = GMSH_TYPES[e]
        if dim_of(nen) == 2:
            f = by_nodes[tuple(sorted(to_idx(nodes, nen)))]
            for p in phys:
                surf.setdefault(names.get((2, p), f"surface_{p}"), []).append(f)
    for name, fs in surf.items():
        m.faces[name] = np.array(fs)
        m.nodes[name] = np.unique(m.faces[name])
    return m


def write_msh(path, m, version="4.1"):
    """gmsh .msh (ASCII) of m: physical volume ALL, one physical surface per string-named face set"""
    gtype = {8: 5, 10: 11, 4: 3, 6: 9}
    order = lambda c: np.asarray(c)[GMSH_ORDER.get(len(c), slice(None))] + 1
    named = [(k, f) for k, f in m.faces.items() if isinstance(k, str)]
    out = ["$MeshFormat", f"{version} 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(named) + 1),
           '3 1 "ALL"'] + [f'2 {i + 2} "{k}"' for i, (k, _) in enumerate(named)] + ["$EndPhysicalNames"]
    elems = [(3, 1, gtype[m.conn.shape[1]], m.conn)] + [(2, i + 2, gtype[f.shape[1]], f) for i, (_, f) in enumerate(named)]
    if version.startswith("2"):
        out += ["$Nodes", str(m.n_nodes)] + [f"{i + 1} {x:.16g} {y:.16g} {z:.16g}" for i, (x, y, z) in enumerate(m.X)]
        out += ["$EndNodes", "$Elements", str(sum(len(c) for *_, c in elems))]
        k = 0
        for dim, tag, t, cs in elems:
            for c in cs:
                k += 1
                out.append(f"{k} {t} 2 {tag} {tag} " + " ".join(map(str, order(c))))
        out.append("$EndElements")
    else:
        out += ["$Entities", f"0 0 {len(named)} 1"]
        out += [f"{i + 2} 0 0 0 0 0 0 1 {i + 2} 0" for i in range(len(named))] + ["1 0 0 0 0 0 0 1 1 0", "$EndEntities"]
        out += ["$Nodes", f"1 {m.n_nodes} 1 {m.n_nodes}", f"3 1 0 {m.n_nodes}"]
        out += [str(i + 1) for i in range(m.n_nodes)] + [f"{x:.16g} {y:.16g} {z:.16g}" for x, y, z in m.X]
        out += ["$EndNodes", "$Elements", f"{len(elems)} {sum(len(c) for *_, c in elems)} 1 {sum(len(c) for *_, c in elems)}"]
        k = 0
        for dim, tag, t, cs in elems:
            out.append(f"{dim} {tag} {t} {len(cs)}")
            for c in cs:
                k += 1
                out.append(f"{k} " + " ".join(map(str, order(c))))
        out.append("$EndElements")
    pathlib.Path(path).write_text("\n".join(out) + "\n")


def box(n=(1, 1, 1), L=(50.0, 50.0, 50.0), perturb=0.0, seed=0, etype="hex8"):
    """Box [0,L0]x[0,L1]x[0,L2]; interior nodes optionally perturbed by perturb * h (hex8)."""
    m = structured(n, lambda s: s * np.array(L), etype)
    if perturb > 0:
        ijk = np.round(m.X / (np.array(L) / np.array(n))).astype(int)
        interior = np.all((ijk > 0) & (ijk < np.array(n)), axis=1)
        h = np.array(L) / np.array(n)
        m.X[interior] += perturb * h * np.random.default_rng(seed).uniform(-1, 1, (interior.sum(), 3))
    return m


def quarter_cylinder(r_i=5.0, t=1.3, L=0.22, n=(8, 60, 4), etype="hex8"):
    """Index axes (r, theta, z); theta in [0, pi/2], cylinder axis = z.

    nodes/faces: (0,-1) inner, (0,1) outer, (1,-1) theta = 0 (y = 0), (1,1) theta = pi/2 (x = 0), (2,+-1) ends.
    """
    def mapping(s):
        r, th, z = r_i + t * s[:, 0], 0.5 * np.pi * s[:, 1], L * s[:, 2]
        return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    return structured(n, mapping, etype)


def cylinder_basis(x):
    """Columns (e_r, e_theta, e_z) at points x, (..., 3, 3)"""
    th = np.arctan2(x[..., 1], x[..., 0])
    c, s, o, i = np.cos(th), np.sin(th), np.zeros_like(th), np.ones_like(th)
    return np.stack([np.stack([c, s, o], -1), np.stack([-s, c, o], -1), np.stack([o, o, i], -1)], -1)


def half_tube(r_i=5.0, t=1.3, L=30.0, n=(3, 24, 30), etype="hex8"):
    """Straight half tube (y >= 0) along z; named nodes/faces: inner, outer, inlet (z = 0), outlet (z = L), sym (y = 0)"""
    def mapping(s):
        r, th = r_i + t * s[:, 0], np.pi * s[:, 1]
        return np.stack([r * np.cos(th), r * np.sin(th), L * s[:, 2]], axis=1)

    m = structured(n, mapping, etype)
    for name, key in (("inner", (0, -1)), ("outer", (0, 1)), ("inlet", (2, -1)), ("outlet", (2, 1))):
        m.nodes[name], m.faces[name] = m.nodes[key], m.faces[key]
    m.nodes["sym"] = np.union1d(m.nodes[(1, -1)], m.nodes[(1, 1)])
    return m


def tube(r_i=5.0, t=1.3, L=30.0, n=(3, 48, 30), etype="hex8"):
    """Full straight tube along z, closed in theta (nodes at theta = 2 pi merged into theta = 0, Mesh.param[:, 1] in
    [0, 1)); named nodes/faces: inner, outer, inlet (z = 0), outlet (z = L)"""
    def mapping(s):
        r, th = r_i + t * s[:, 0], 2 * np.pi * s[:, 1]
        return np.stack([r * np.cos(th), r * np.sin(th), L * s[:, 2]], axis=1)

    m = structured(n, mapping, etype)
    key = lambda i: (round(float(m.param[i, 0]), 9), round(float(m.param[i, 2]), 9))
    seam = {key(i): i for i in m.nodes[(1, -1)]}
    new = np.arange(m.n_nodes)
    new[m.nodes[(1, 1)]] = [seam[key(i)] for i in m.nodes[(1, 1)]]
    used = np.unique(new)
    order = np.full(m.n_nodes, -1)
    order[used] = np.arange(len(used))
    remap = lambda a: order[new[a]]
    m.X, m.param, m.conn = m.X[used], m.param[used], remap(m.conn)
    for k in ((1, -1), (1, 1)):
        m.nodes.pop(k)
        m.faces.pop(k)
    m.nodes = {k: np.unique(remap(v)) for k, v in m.nodes.items()}
    m.faces = {k: remap(v) for k, v in m.faces.items()}
    for name, k in (("inner", (0, -1)), ("outer", (0, 1)), ("inlet", (2, -1)), ("outlet", (2, 1))):
        m.nodes[name], m.faces[name] = m.nodes[k], m.faces[k]
    return m


def bent_tube(r_i=5.0, t=1.3, R_c=20.0, L_in=10.0, L_out=10.0, stenosis=0.25, width=0.08, n=(3, 24, 40),
              etype="hex8", intima=None):
    """Half tube (y >= 0) along a centerline: straight z (inlet at z = 0), 90 deg arc of radius R_c, straight x.

    r_inner(l) = r_i (1 - stenosis exp(-((l/L - 1/2)/width)^2)),  l = arc length, L = total length
    intima=(t_i, n_i): outer radius r_i + t_i + t constant, the first n_i of n[0] layers are an intima of thickness
    t_i + r_i - r_inner(l) (plaque), the others the media of thickness t; else the whole wall t is indented
    named nodes/faces: inner, outer, inlet (z = 0), outlet (x = R_c + L_out), sym (y = 0)
    """
    L = L_in + 0.5 * np.pi * R_c + L_out

    def frame(l):
        phi = np.clip((l - L_in) / R_c, 0.0, 0.5 * np.pi)
        after = np.maximum(l - L_in - 0.5 * np.pi * R_c, 0.0)
        before = np.minimum(l - L_in, 0.0)
        c = np.stack([R_c * (1 - np.cos(phi)) + after, 0 * l, L_in + before + R_c * np.sin(phi)], -1)
        n1 = np.stack([np.cos(phi), 0 * l, -np.sin(phi)], -1)
        return c, n1

    def mapping(s):
        l = L * s[:, 2]
        bump = r_i * stenosis * np.exp(-((s[:, 2] - 0.5) / width) ** 2)
        if intima is None:
            r = r_i - bump + t * s[:, 0]
        else:
            t_i, f = intima[0], intima[1] / n[0]
            r = np.where(s[:, 0] < f, r_i + t_i - (t_i + bump) * (1 - s[:, 0] / f), r_i + t_i + t * (s[:, 0] - f) / (1 - f))
        th = np.pi * s[:, 1]
        c, n1 = frame(l)
        return c + r[:, None] * (np.cos(th)[:, None] * n1 + np.sin(th)[:, None] * np.array([0.0, 1.0, 0.0]))

    m = structured(n, mapping, etype)
    for name, key in (("inner", (0, -1)), ("outer", (0, 1)), ("inlet", (2, -1)), ("outlet", (2, 1))):
        m.nodes[name], m.faces[name] = m.nodes[key], m.faces[key]
    m.nodes["sym"] = np.union1d(m.nodes[(1, -1)], m.nodes[(1, 1)])
    return m


def carotid_bifurcation(D=6.3, angles=(25.0, 25.0), sinus=1.11, ica=0.71, eca=0.57, s_sinus=(0.3, 0.8, 1.8),
                        s_cca=0.5, lengths=(3.0, 5.0, 3.0), cuffs=(0.6, 1.5, 1.4), blend=(0.25, 0.08), kappa=0.26,
                        kappa_junction=None, n_q=6, n_seam=8, n_r=2, n_smooth=20, thickness="curvature", half=False,
                        fillet=None):
    """Idealized planar carotid bifurcation, hex8 wall; bifurcation plane y = 0, centerlines meet at the origin,
    CCA along z (z < 0), ICA / ECA at angles (deg) from z towards +x / -x.

    Diameters / D (Bharadvaj et al. 1982, Ku et al. 1985): sinus, distal ICA, ECA. Radius along each axis, s = distance
    from the origin (in D): ICA 0.9 R -> sinus R on [0, s1], sinus until s2, -> ica R at s3; ECA eca R; the CCA
    cylinder (radius R) reaches s_cca beyond the origin, so the ECA tapers out of it without a waist, and the ICA
    starts inside it (no coincident surfaces).
    Lumen F = smin(d_cca, smin(d_ica, d_eca; k_0); k_1) < 0, d_j = |x - s_j e_j| - r_j(s_j) (s_j clipped to the
    branch, spherical caps), smin quadratic with (k_0, k_1) = blend D: flow divider radius from k_0, small k_1 since
    the CCA and ICA are nearly tangent at the origin (smin bulges by k / 4 where surfaces coincide).
    fillet (f_0, f_1): CAD junction instead (no bulge): CCA ending in a half-ball of radius R at the origin, ICA (sinus
    R, -> ica R on [s2, s3]) and ECA axes from the points that put their outer wall lines on the CCA rim, lumen
    tube_field with rounds rho = f_0 (R + r_b) / 2 on CCA / half-ball - branch, f_1 (r_i + r_e) / 2 on the flow
    divider (s from those points; the taper added where the ICA is the nearest tube).
    Wall thickness kappa r_local along the lumen normal (averaged n_smooth times over the surface neighbours, so the
    offset does not fold at the flow divider), n_r layers; thickness "radius": r_local = radius of the nearest tube,
    "curvature": Laplace radius r_local = 1 / div n (mean curvature; = r in a tube), in [0.75, 3] x the tube radius,
    smoothed: the flat front / back of the junction and the flow divider get a thicker wall, so a membrane at the
    homeostatic stress of the tubes can carry the pressure there; kappa_junction: kappa on the junction patches instead
    (smooth transition over n_smooth neighbour averages).
    Surface: three tubes (4 n_q cells around, cuffs at s = cuffs D) joined per side (y > 0, y < 0) by three patches of
    (n_q + n_seam)^2 quads meeting at one point; seams along the in-plane lumen outline (lateral walls, flow divider).
    named nodes/faces: inner, outer, inlet (CCA), ica, eca, sym (half: y >= 0 only); elsets cca, ica, eca, junction
    """
    R, (k0, k1) = 0.5 * D, (v * D for v in blend)
    a, b = np.radians(angles)
    sm = lambda s, lo, hi: (lambda t: t * t * (3 - 2 * t))(np.clip((s - lo) / (hi - lo), 0.0, 1.0))
    s1, s2, s3 = (v * D for v in s_sinus)
    cad = fillet is not None
    e_c, e_i, e_e = np.array([0.0, 0, 1]), np.array([np.sin(a), 0, np.cos(a)]), np.array([-np.sin(b), 0, np.cos(b)])
    r_ica = (lambda s: sinus * R + (ica - sinus) * R * sm(s, s2, s3)) if cad else \
        (lambda s: 0.9 * R + (sinus - 0.9) * R * sm(s, 0, s1) + (ica - sinus) * R * sm(s, s2, s3))
    r_j = np.array([R, sinus * R, eca * R])
    origin = np.zeros((3, 3))
    if cad:
        for j, e, rim in ((1, e_i, np.array([R, 0, 0])), (2, e_e, np.array([-R, 0, 0]))):
            v = rim - (rim @ e) * e
            origin[j] = rim - r_j[j] * v / np.linalg.norm(v)
    branches = [  # axis origin, axis, frame e1 (phi = 0), clip of s, radius
        (origin[0], e_c, np.array([1.0, 0, 0]), (-np.inf, 0.0 if cad else s_cca * D), lambda s: R + 0 * s),
        (origin[1], e_i, np.array([np.cos(a), 0, -np.sin(a)]), (0.0, np.inf), r_ica),
        (origin[2], e_e, np.array([np.cos(b), 0, np.sin(b)]), (0.0, np.inf), lambda s: eca * R + 0 * s)]
    ey = np.array([0.0, 1, 0])

    def dist(x):
        out = []
        for o, e, _, clip, r in branches:
            s = np.clip((x - o) @ e, *clip)
            out.append(np.linalg.norm(x - o - s[:, None] * e, axis=1) - r(s))
        return out

    def smin(p, q, k):
        h = np.maximum(k - np.abs(p - q), 0.0) / k
        return np.minimum(p, q) - 0.25 * k * h * h

    if cad:
        rho = np.zeros((3, 3))
        rho[0, 1:] = rho[1:, 0] = fillet[0] * (R + r_j[1:]) / 2
        rho[1, 2] = rho[2, 1] = fillet[1] * r_j[1:].mean()
        dirs = np.array([-e_c, e_i, e_e])
        F_cad = tube_field(origin, origin + (np.array(lengths) * D + R + 2 * rho.max())[:, None] * dirs, r_j, rho,
                           1e-6 * R)[0]
        F = lambda x: (lambda d: F_cad(x) + np.where(np.argmin(d, 0) == 1, sinus * R - r_ica(
            np.maximum((x - origin[1]) @ e_i, 0.0)), 0.0))(np.stack(dist(x)))
    else:
        F = lambda x: (lambda d: smin(d[0], smin(d[1], d[2], k0), k1))(dist(x))

    def grad(x, eps=1e-6 * D):
        return np.stack([(F(x + eps * e) - F(x - eps * e)) / (2 * eps) for e in np.eye(3)], -1)

    def project(x, it=12):
        x = np.array(x, dtype=float).reshape(-1, 3)
        for _ in range(it):
            g = grad(x)
            x = x - (F(x) / np.sum(g * g, -1))[:, None] * g
        return x

    def ring(j, s, n):
        o, e, e1, _, r = branches[j]
        ph = np.linspace(0.0, np.pi, n + 1)
        return project(o + s * e + r(np.asarray(s)) * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * ey))

    s_cuff = (-cuffs[0] * D, cuffs[1] * D, cuffs[2] * D)
    s_end = (-lengths[0] * D, lengths[1] * D, lengths[2] * D)
    cuff = [ring(j, s_cuff[j], 2 * n_q) for j in range(3)]
    h_el = 0.5 * np.pi * R / n_q

    def seam(p, q, e_stop, t0, step=0.004 * D):
        """in-plane lumen outline from p to the plane through q normal to e_stop, 2 n_seam + 1 points"""
        pts, t = [p], t0
        while (pts[-1] - q) @ e_stop < 0 and len(pts) < 100000:
            g = grad(pts[-1][None])[0]
            n = np.array([g[2], 0.0, -g[0]])
            t = np.sign(n @ t) * n / np.linalg.norm(n)
            pts.append(project(pts[-1] + step * t)[0])
        pts[-1] = q
        pts = np.array(pts)
        l = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        out = project(np.stack([np.interp(np.linspace(0, l[-1], 2 * n_seam + 1), l, pts[:, i]) for i in range(3)], -1))
        out[0], out[-1] = p, q
        return out

    seams =[seam(cuff[0][0], cuff[1][0], e_i, e_c),
             seam(cuff[1][-1], cuff[2][0], e_e, -e_i),
             seam(cuff[2][-1], cuff[0][-1], -e_c, -e_e)]
    mid = [c[n_q] for c in cuff]
    base = np.mean(mid, 0) * np.array([1.0, 0, 1])
    lo, hi = 0.0, 2 * D
    for _ in range(60):
        lo, hi = ((lo + hi) / 2, hi) if F((base + (lo + hi) / 2 * ey)[None])[0] < 0 else (lo, (lo + hi) / 2)
    center = base + lo * ey
    M = n_q + n_seam

    def coons(side_a, side_b):
        """corners center, side_a[0], side_a[-1] = side_b[-1], side_b[0]; sides of M + 1 points"""
        u = np.linspace(0, 1, M + 1)
        B0 = center + u[:, None] * (side_a[0] - center)
        L0 = center + u[:, None] * (side_b[0] - center)
        U, V = np.meshgrid(u, u, indexing="ij")
        P = ((1 - V)[..., None] * B0[:, None] + V[..., None] * side_b[:, None]
             + (1 - U)[..., None] * L0[None] + U[..., None] * side_a[None]
             - ((1 - U) * (1 - V))[..., None] * center - (U * (1 - V))[..., None] * side_a[0]
             - ((1 - U) * V)[..., None] * side_b[0] - (U * V)[..., None] * side_a[-1])
        return P

    patches = [coons(np.concatenate([cuff[0][n_q::-1], seams[0][1:n_seam + 1]]),
                     np.concatenate([cuff[1][n_q::-1], seams[0][:n_seam - 1:-1][1:]])),
               coons(np.concatenate([cuff[1][n_q:], seams[1][1:n_seam + 1]]),
                     np.concatenate([cuff[2][n_q::-1], seams[1][:n_seam - 1:-1][1:]])),
               coons(np.concatenate([cuff[2][n_q:], seams[2][1:n_seam + 1]]),
                     np.concatenate([cuff[0][n_q:], seams[2][:n_seam - 1:-1][1:]]))]

    X, quads, region, free = [], [], [], []

    def add(P, reg, interior):
        """structured (n_i + 1, n_j + 1, 3) grid of surface nodes"""
        ni, nj = P.shape[0] - 1, P.shape[1] - 1
        idx = sum(len(x) for x in X) + np.arange((ni + 1) * (nj + 1)).reshape(ni + 1, nj + 1)
        X.append(P.reshape(-1, 3))
        free.append(np.full((ni + 1) * (nj + 1), interior))
        quads.append(np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4))
        region.extend([reg] * ni * nj)

    ends = []
    for j in range(3):
        n_ax = max(int(np.ceil(abs(s_end[j] - s_cuff[j]) / h_el)), 1)
        s = np.linspace(s_cuff[j], s_end[j], n_ax + 1)
        P = np.stack([cuff[j]] + [ring(j, v, 2 * n_q) for v in s[1:]])
        add(P, j, False)
        ends.append(P[-1])
    for P in patches:
        add(P, 3, True)
    X, quads, free = np.concatenate(X), np.concatenate(quads), np.concatenate(free)

    tag = lambda x: np.round(x / (1e-9 * D)).astype(np.int64)

    def merge(X, quads, free):
        """coincident nodes merged (shared lines are built from the same arrays)"""
        _, first, inv = np.unique(tag(X), axis=0, return_index=True, return_inverse=True)
        f = np.zeros(len(first), bool)
        np.logical_or.at(f, inv.ravel(), free)
        return X[first], inv.ravel()[quads], f

    X, quads, free = merge(X, quads, free)
    key = {tuple(v): i for i, v in enumerate(tag(X))}
    lookup = lambda pts: np.array([key[tuple(v)] for v in tag(pts)])
    for line in seams + cuff:
        free[lookup(line)] = False
    X[free] = project(X[free])
    def average(quads, n):
        """mean over the edge neighbours"""
        e = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
        A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), (n, n)).tocsr()
        A.data[:] = 1.0
        return sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A

    A = average(quads, len(X))
    for _ in range(300):
        X[free] = project(A[free] @ X, it=3)

    region = np.array(region)
    if not half:
        Xm = X * np.array([1.0, -1, 1])
        X, quads, _ = merge(np.concatenate([X, Xm]), np.concatenate([quads, quads[:, ::-1] + len(X)]),
                            np.zeros(2 * len(X), bool))
        region = np.concatenate([region, region])
        ends = [np.concatenate([p, p * np.array([1.0, -1, 1])]) for p in ends]

    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    n, A = unit(grad(X)), average(quads, len(X))
    plane = np.abs(X[:, 1]) < 1e-9 * D
    for _ in range(n_smooth):
        n = unit(n + A @ n)
        n[plane, 1] = 0.0
    d = np.stack(dist(X), -1)
    w = np.exp(-(d - d.min(1, keepdims=True)) / (0.1 * D))
    r_loc = np.sum(w * np.stack([np.abs(r(np.clip((X - o) @ e, *clip))) for o, e, _, clip, r in branches], -1),
                   1) / w.sum(1)
    if thickness == "curvature":
        e_h = 1e-3 * D
        div = sum((unit(grad(X + e_h * e))[:, i] - unit(grad(X - e_h * e))[:, i]) / (2 * e_h)
                  for i, e in enumerate(np.eye(3)))
        r_loc = np.clip(1.0 / np.maximum(div, 1e-12), 0.75 * r_loc, 3.0 * r_loc)
        for _ in range(n_smooth // 4):
            r_loc = 0.5 * (r_loc + A @ r_loc)
    x = X[quads]
    flip = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 3] - x[:, 0]), n[quads].mean(1)) < 0
    quads[flip] = quads[flip][:, ::-1]
    ns = len(X)
    k_loc = np.full(len(X), kappa)
    if kappa_junction is not None:
        k_loc[np.unique(quads[region == 3])] = kappa_junction
        for _ in range(n_smooth):
            k_loc = 0.5 * (k_loc + A @ k_loc)
    X3 = np.concatenate([X + (j / n_r * k_loc * r_loc)[:, None] * n for j in range(n_r + 1)])
    conn = np.concatenate([np.concatenate([quads + j * ns, quads + (j + 1) * ns], 1) for j in range(n_r)])
    m = Mesh(X=X3, conn=conn)
    m.elsets = {name: np.flatnonzero(np.tile(region, n_r) == j) for j, name in enumerate(("cca", "ica", "eca", "junction"))}
    key = {tuple(v): i for i, v in enumerate(tag(X))}
    surf = {name: np.unique(lookup(p)) for name, p in zip(("inlet", "ica", "eca"), ends)}
    m.nodes["inner"], m.nodes["outer"] = np.arange(ns), n_r * ns + np.arange(ns)
    for name, s in surf.items():
        m.nodes[name] = np.concatenate([s + j * ns for j in range(n_r + 1)])
    if half:
        m.nodes["sym"] = np.flatnonzero(np.abs(X3[:, 1]) < 1e-9 * D)
    elems, _, faces = boundary_faces(m)
    for name in ("inner", "outer", "inlet", "ica", "eca") + (("sym",) if half else ()):
        on = np.all(np.isin(faces, m.nodes[name]), 1)
        m.faces[name] = faces[on]
    return m


def tube_field(A, B, r, rho, tol):
    """Signed distance-like field of straight tubes (axis A_i -> B_i, radii r_i; rounded at A_i by a half-ball, flat
    at B_i) with a round of constant radius rho_ij (0: none) on the intersection line of tubes i and j only, as a CAD
    round on an edge: a ball rolls along tubes i and j, what it cannot reach is filled:
        d_i: exact distance to tube i
        H_ij(x) = max_y (rho_ij - |x - y|)   over the ball centres y with min(d_i(y), d_j(y)) >= rho_ij
        F = min(min_k d_k, min_ij H_ij)
    The maximum over y is taken over its candidates: x itself, the projections along the gradient onto the offset
    tubes d_i = rho_ij, d_j = rho_ij and the nearest point on the ball-centre line d_i = d_j = rho_ij (Newton,
    accepted within tol); only where both tubes are within rho_ij (the round lies there). F = d_i away from the
    intersection lines, tangent rounds on them, no material anywhere else.
    Returns F(x), tubes(x) -> (d (n, m), unit gradients (n, m, 3))"""
    A, B, r = (np.asarray(v, dtype=float) for v in (A, B, r))
    m = len(r)
    pairs = np.array(list(itertools.combinations(range(m), 2)))
    rho_pair = np.broadcast_to(np.asarray(rho, dtype=float), (m, m))[pairs[:, 0], pairs[:, 1]]
    L = np.linalg.norm(B - A, axis=1)
    axis_dir = (B - A) / L[:, None]
    tiny = 1e-9 * r.min()

    def tubes(x):
        q = x[:, None] - A
        t = np.einsum("nij,ij->ni", q, axis_dir)
        v = q - t[..., None] * axis_dir
        rad = np.maximum(np.linalg.norm(v, axis=2), tiny)
        u, a, b = v / rad[..., None], rad - r, t - L
        hyp = np.maximum(np.hypot(np.maximum(a, 0.0), b), tiny)
        d = np.where(b > 0, hyp, np.maximum(a, b))
        n = np.where((b > 0)[..., None], (np.maximum(a, 0.0)[..., None] * u + b[..., None] * axis_dir) / hyp[..., None],
                     np.where((a >= b)[..., None], u, axis_dir))
        qn = np.maximum(np.linalg.norm(q, axis=2), tiny)
        cap = t < 0
        return np.where(cap, qn - r, d), np.where(cap[..., None], q / qn[..., None], n)

    def F(x_all):
        d_all, n_all = tubes(x_all)
        out = d_all.min(1)
        for (i, j), rp in zip(pairs, rho_pair):
            near = (rp > 0) & (np.maximum(d_all[:, i], d_all[:, j]) < rp)
            x, d, n = x_all[near], d_all[near], n_all[near]
            G = np.minimum(d[:, i], d[:, j])
            best = np.where(G >= rp, rp, -np.inf)
            for a, b in ((i, j), (j, i)):
                y = x + (rp - d[:, a])[:, None] * n[:, a]
                ok = tubes(y)[0][:, b] >= rp - tol
                best = np.where(ok, np.maximum(best, rp - np.linalg.norm(x - y, axis=1)), best)
            y = x + (rp - d[:, i])[:, None] * n[:, i] + (rp - d[:, j])[:, None] * n[:, j]
            for _ in range(12):
                dy, ny = tubes(y)
                J = ny[:, [i, j]]
                Jt = np.swapaxes(J, 1, 2)
                y = y - (Jt @ np.linalg.solve(J @ Jt + 1e-12 * np.eye(2), (dy[:, [i, j]] - rp)[..., None]))[..., 0]
                tau = np.cross(J[:, 0], J[:, 1])
                tau /= np.maximum(np.linalg.norm(tau, axis=1, keepdims=True), 1e-12)
                y = y + np.sum((x - y) * tau, 1)[:, None] * tau
            ok = np.abs(tubes(y)[0][:, [i, j]] - rp).max(1) < tol
            best = np.where(ok, np.maximum(best, rp - np.linalg.norm(x - y, axis=1)), best)
            out[near] = np.minimum(out[near], np.where(np.isfinite(best), best, G))
        return out

    return F, tubes


def tube_junction(A, B, r, rho, voxel=None, smooth=5):
    """Watertight triangle surface F = 0 of tube_field (tubes A_i -> B_i, radii r_i, rounds rho_ij).
    Zero set by marching tetrahedra on the voxel grid over the bounding box (each cell split into 6
    tetrahedra along its main diagonal, consistent across cells: watertight and manifold), vertices shared per grid
    edge; smooth relaxation passes (vertex -> mean of its neighbours) against slivers, each followed by a projection
    onto F = 0 (Newton), so the surface does not shrink.
    Returns vertices, triangles (outward normals), F"""
    A, B, r = (np.asarray(v, dtype=float) for v in (A, B, r))
    voxel = voxel or r.min() / 8
    rho_max = np.asarray(rho, dtype=float).max()
    F, tubes = tube_field(A, B, r, rho, 1e-3 * voxel)
    lo = np.minimum(A, B).min(0) - r.max() - 2 * voxel
    hi = np.maximum(A, B).max(0) + r.max() + 2 * voxel
    axes = [np.arange(l, h + voxel, voxel) for l, h in zip(lo, hi)]
    shape = tuple(len(v) for v in axes)
    Xg = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    g = tubes(Xg)[0].min(1)
    near = (g > -2 * voxel) & (g < max(rho_max, 2 * voxel))
    g[near] = F(Xg[near])
    g[g == 0] = 1e-12 * voxel
    idx = np.arange(g.size).reshape(shape)
    corner = {v: idx[v[0]:shape[0] - 1 + v[0], v[1]:shape[1] - 1 + v[1], v[2]:shape[2] - 1 + v[2]].ravel()
              for v in itertools.product((0, 1), repeat=3)}
    vals = np.stack([g[c8] for c8 in corner.values()], 1)
    cut = (vals.min(1) < 0) & (vals.max(1) > 0)
    tets = []
    for perm in itertools.permutations(range(3)):
        path = [(0, 0, 0)]
        for ax in perm:
            path.append(tuple(int(path[-1][i] or i == ax) for i in range(3)))
        tets.append(np.stack([corner[v][cut] for v in path], 1))
    tets = np.concatenate(tets)
    code = (g[tets] < 0) @ np.array([1, 2, 4, 8])
    P, Q = [], []
    for mask in range(1, 15):
        ins = [i for i in range(4) if mask >> i & 1]
        out = [i for i in range(4) if not mask >> i & 1]
        if len(ins) == 2:
            tris = [[(ins[0], out[0]), (ins[0], out[1]), (ins[1], out[1])],
                    [(ins[0], out[0]), (ins[1], out[1]), (ins[1], out[0])]]
        else:
            tris = [[(ins[0], o) for o in out]] if len(ins) == 1 else [[(i, out[0]) for i in ins]]
        sel = tets[code == mask]
        for tri in tris:
            P.append(sel[:, [p for p, _ in tri]])
            Q.append(sel[:, [q for _, q in tri]])
    P, Q = np.concatenate(P), np.concatenate(Q)
    _, first, T = np.unique(P.ravel() * g.size + Q.ravel(), return_index=True, return_inverse=True)
    p, q = P.ravel()[first], Q.ravel()[first]
    V = Xg[p] + (g[p] / (g[p] - g[q]))[:, None] * (Xg[q] - Xg[p])
    T = T.reshape(-1, 3)
    flip = np.einsum("ij,ij->i", np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]),
                     Xg[Q[:, 0]] - Xg[P[:, 0]]) < 0
    T[flip] = T[flip][:, ::-1]
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    Adj = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), (len(V), len(V)))
    Adj = Adj.tocsr()
    Adj.data[:] = 1.0
    Adj = sp.diags(1.0 / np.asarray(Adj.sum(1)).ravel()) @ Adj
    eps = 1e-4 * voxel
    for it in range(smooth + 1):
        V = Adj @ V if it else V
        for _ in range(3):
            n = np.stack([(F(V + eps * e) - F(V - eps * e)) / (2 * eps) for e in np.eye(3)], -1)
            V = V - (F(V) / np.sum(n * n, 1))[:, None] * n
    return V, T, F


def _data_array(name, v, vtk_type="Float64"):
    """Base64 binary DataArray; (n, 3, 3) is written as a 9-component tensor"""
    dtype = {"Float64": np.float64, "Int64": np.int64, "UInt8": np.uint8}[vtk_type]
    v = np.ascontiguousarray(v, dtype=dtype)
    ncomp = 1 if v.ndim == 1 else int(np.prod(v.shape[1:]))
    raw = v.tobytes()
    data = base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode()
    return (f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="binary">'
            f"{data}</DataArray>")


def write_vtu(path, X, conn, point_data=None, cell_data=None):
    """VTK XML unstructured grid (hex8: VTK_HEXAHEDRON, tet10: VTK_QUADRATIC_TETRA, tet4: VTK_TETRA, same node order;
    3 nodes: VTK_TRIANGLE surface)"""
    vtk_type, n_nodes = (5, 3) if conn.shape[1] == 3 else (element_for(conn.shape[1]).vtk, conn.shape[1])
    n, e = len(X), len(conn)
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">',
           "<UnstructuredGrid>", f'<Piece NumberOfPoints="{n}" NumberOfCells="{e}">',
           "<Points>", _data_array("Points", X), "</Points>",
           "<Cells>", _data_array("connectivity", np.ravel(conn), "Int64"),
           _data_array("offsets", n_nodes * np.arange(1, e + 1), "Int64"),
           _data_array("types", np.full(e, vtk_type), "UInt8"), "</Cells>"]
    for tag, data in (("PointData", point_data), ("CellData", cell_data)):
        out += [f"<{tag}>"] + [_data_array(k, v) for k, v in (data or {}).items()] + [f"</{tag}>"]
    out += ["</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    pathlib.Path(path).write_text("\n".join(out) + "\n")


def read_vtu(path):
    """(X, conn, point_data, cell_data) of a file written by write_vtu"""
    text = pathlib.Path(path).read_text()
    dtypes = {"Float64": np.float64, "Int64": np.int64, "UInt8": np.uint8}

    def arrays(tag):
        block = text.split(f"<{tag}>")[1].split(f"</{tag}>")[0] if f"<{tag}>" in text else ""
        out = {}
        for vtk_type, name, ncomp, data in re.findall(
                r'<DataArray type="(\w+)" Name="(\w+)" NumberOfComponents="(\d+)" format="binary">([^<]*)</DataArray>',
                block):
            raw = base64.b64decode(data)
            v = np.frombuffer(raw[4:4 + int(np.frombuffer(raw[:4], np.uint32)[0])], dtypes[vtk_type])
            out[name] = v.reshape(-1, int(ncomp)) if int(ncomp) > 1 else v
        return out

    cells = arrays("Cells")
    return (arrays("Points")["Points"], cells["connectivity"].reshape(len(cells["offsets"]), -1),
            arrays("PointData"), arrays("CellData"))


def read_pvd(pvd):
    """[(t, path of the .vtu)] of a ParaView collection"""
    pvd = pathlib.Path(pvd)
    return [(float(t), pvd.parent / f) for t, f in re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', pvd.read_text())]


class VTKSeries:
    """ParaView time series: <stem>/<k>.vtu per call and <stem>.pvd indexing them by time (rewritten each call);
    mesh: its named sets to <stem>.inp (FEM Sets filter of fem/paraview_plugin.py)"""

    def __init__(self, pvd, mesh=None):
        self.pvd = pathlib.Path(pvd)
        self.dir = self.pvd.with_suffix("")
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.glob("*.vtu"):
            f.unlink()
        self.entries = []
        if mesh is not None:
            write_inp(self.pvd.with_suffix(".inp"), mesh)

    def write(self, t, X, conn, point_data=None, cell_data=None):
        """point_data with "u": points at the deformed X + u, the reference X as point data X_ref"""
        if point_data and "u" in point_data:
            point_data = {"X_ref": np.asarray(X, dtype=float)} | dict(point_data)
            X = point_data["X_ref"] + np.asarray(point_data["u"]).reshape(point_data["X_ref"].shape)
        name = f"{self.dir.name}/{len(self.entries):04d}.vtu"
        write_vtu(self.pvd.parent / name, X, conn, point_data, cell_data)
        self.entries.append((t, name))
        rows = [f'<DataSet timestep="{t:g}" part="0" file="{f}"/>' for t, f in self.entries]
        self.pvd.write_text("\n".join(['<?xml version="1.0"?>', '<VTKFile type="Collection" version="1.0">',
                                       "<Collection>", *rows, "</Collection>", "</VTKFile>"]) + "\n")
