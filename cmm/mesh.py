"""Meshes of hex8 or tet10: structured generators, boundary faces, Abaqus .inp and gmsh .msh I/O, legacy VTK."""

import itertools
import pathlib
from dataclasses import dataclass, field

import numpy as np

from elements import HEX_CORNERS, ELEMENT_TYPES, element_for, face_for

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
    tet10: 6 Kuhn tets per cell, midside nodes mapped from index space (curved edges follow the mapping)."""
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
        grid, inv = np.unique(np.concatenate([corners, mids], 1).reshape(-1, 3), axis=0, return_inverse=True)
        conn = inv.reshape(-1, 10)
        top = 2 * n
    X = mapping(grid / top.astype(float))
    if etype == "tet10":
        x = X[conn[:, :4]]
        neg = np.linalg.det(np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], 1)) < 0
        conn[neg] = conn[neg][:, TET10_SWAP]

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


def faces_on(m, node_set):
    """Outward boundary faces with all nodes in node_set"""
    _, _, faces = boundary_faces(m)
    return faces[np.all(np.isin(faces, node_set), axis=1)]


def _ids(tokens):
    return [int(t) for t in tokens if t.strip()]


def read_inp(path):
    """Abaqus .inp (single part, C3D8* or C3D10*) -> Mesh with *Nset, *Elset and element-based *Surface by name."""
    node_id, X, elem_id, conn = [], [], [], []
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
                    raise ValueError(f"element type {etype} not supported (one of C3D8*, C3D10*)")
                nen = el.n_nodes
            if block == "surface":
                surfaces[opts["name"]] = []
            continue
        tok = [t.strip() for t in line.rstrip(",").split(",")]
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


def bent_tube(r_i=5.0, t=1.3, R_c=20.0, L_in=10.0, L_out=10.0, stenosis=0.25, width=0.08, n=(3, 24, 40),
              etype="hex8"):
    """Half tube (y >= 0) along a centerline: straight z (inlet at z = 0), 90 deg arc of radius R_c, straight x.

    r_inner(l) = r_i (1 - stenosis exp(-((l/L - 1/2)/width)^2)),  l = arc length, L = total length
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
        r = r_i * (1 - stenosis * np.exp(-((s[:, 2] - 0.5) / width) ** 2)) + t * s[:, 0]
        th = np.pi * s[:, 1]
        c, n1 = frame(l)
        return c + r[:, None] * (np.cos(th)[:, None] * n1 + np.sin(th)[:, None] * np.array([0.0, 1.0, 0.0]))

    m = structured(n, mapping, etype)
    for name, key in (("inner", (0, -1)), ("outer", (0, 1)), ("inlet", (2, -1)), ("outlet", (2, 1))):
        m.nodes[name], m.faces[name] = m.nodes[key], m.faces[key]
    m.nodes["sym"] = np.union1d(m.nodes[(1, -1)], m.nodes[(1, 1)])
    return m


def write_vtk(path, X, conn, point_data=None, cell_data=None):
    """Legacy ASCII VTK unstructured grid (hex8: VTK_HEXAHEDRON, tet10: VTK_QUADRATIC_TETRA, same node order)"""
    n, e = len(X), len(conn)
    el = element_for(conn.shape[1])
    out = ["# vtk DataFile Version 3.0", "cmm", "ASCII", "DATASET UNSTRUCTURED_GRID", f"POINTS {n} double"]
    out += [" ".join(f"{v:.10g}" for v in x) for x in X]
    out += [f"CELLS {e} {(el.n_nodes + 1) * e}"] + [f"{el.n_nodes} " + " ".join(map(str, c)) for c in conn]
    out += [f"CELL_TYPES {e}"] + [str(el.vtk)] * e

    def arrays(data):
        lines = []
        for name, v in (data or {}).items():
            v = np.asarray(v, dtype=float)
            head = f"VECTORS {name} double" if v.ndim == 2 else f"SCALARS {name} double 1\nLOOKUP_TABLE default"
            lines += [head] + [" ".join(f"{a:.10g}" for a in np.atleast_1d(r)) for r in v]
        return lines

    if point_data:
        out += [f"POINT_DATA {n}"] + arrays(point_data)
    if cell_data:
        out += [f"CELL_DATA {e}"] + arrays(cell_data)
    pathlib.Path(path).write_text("\n".join(out) + "\n")
