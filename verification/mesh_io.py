"""Mesh file round trips (Abaqus .inp, gmsh .msh 4.1 / 2.2) of a bent tube: nodes, connectivity, named surfaces.
Run: python -m verification.mesh_io
"""

import pathlib
import tempfile

import numpy as np

from fem import mesh as meshlib

OUT = pathlib.Path(__file__).parent / "results"
NAMES = ("inner", "outer", "inlet", "outlet")
FORMATS = (("inp", meshlib.write_inp, meshlib.read_inp),
           ("msh 4.1", lambda q, m: meshlib.write_msh(q, m, "4.1"), meshlib.read_msh),
           ("msh 2.2", lambda q, m: meshlib.write_msh(q, m, "2.2"), meshlib.read_msh))


def same_mesh(m, src):
    ok = np.abs(m.X - src.X).max() < 1e-9 and np.array_equal(m.conn, src.conn)
    for k in NAMES:
        ok &= np.array_equal(np.unique(np.sort(m.faces[k], 1), axis=0), np.unique(np.sort(src.faces[k], 1), axis=0))
    return bool(ok)


def main():
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    log("Mesh I/O round trips, bent tube: coordinates, connectivity, surfaces " + ", ".join(NAMES))
    ok_all = True
    with tempfile.TemporaryDirectory() as d:
        for etype, n in (("hex8", (2, 6, 8)), ("tet10", (1, 6, 8))):
            src = meshlib.bent_tube(n=n, etype=etype)
            for label, write, read in FORMATS:
                path = pathlib.Path(d) / "mesh"
                write(path, src)
                m = read(path)
                ok = same_mesh(m, src)
                ok_all &= ok
                log(f"  {etype:5s} {label:8s} {m.n_elem} elements, faces {[len(m.faces[k]) for k in NAMES]}  "
                    f"{'PASS' if ok else 'FAIL'}")
    log("ALL PASS" if ok_all else "FAILURES")
    OUT.mkdir(exist_ok=True)
    (OUT / "mesh_io.txt").write_text("\n".join(lines) + "\n")
    return ok_all


if __name__ == "__main__":
    main()
