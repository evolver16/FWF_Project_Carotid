"""Shared by the studies: output files with archiving, logging, csv, deformed surface areas."""

import csv
import pathlib
import time

import jax.numpy as jnp
import numpy as np

RESULTS = pathlib.Path(__file__).parent.parent / "results"


def outputs(study, *names):
    """results/<study>/<name>; an existing file (a .pvd with its step folder) moves to results/_archive/<study>/,
    stamped with its modification time"""
    d = RESULTS / study
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / name for name in names]
    for p in paths:
        if not p.exists():
            continue
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(p.stat().st_mtime))
        a = RESULTS / "_archive" / study
        if p.suffix == ".pvd":
            a = a / f"{p.stem}_{stamp}"
            a.mkdir(parents=True, exist_ok=True)
            if p.with_suffix("").is_dir():
                p.with_suffix("").rename(a / p.stem)
            p.rename(a / p.name)
        else:
            a.mkdir(parents=True, exist_ok=True)
            p.rename(a / f"{p.stem}_{stamp}{p.suffix}")
    return paths


class Log:
    """Prints and collects lines; save(path) writes them"""

    def __init__(self):
        self.lines = []

    def __call__(self, text):
        print(text, flush=True)
        self.lines.append(text)

    def save(self, path):
        path.write_text("\n".join(self.lines) + "\n")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def surface_area(m, u, faces):
    """Deformed area of quad4 or tri6 faces by their Gauss rule"""
    from fem.elements import face_for
    face = face_for(faces.shape[1])
    dN = jnp.asarray(np.stack([face.dN(g) for g in face.gauss]))
    x = (jnp.asarray(m.X) + jnp.asarray(u).reshape(-1, 3))[faces]
    a1, a2 = jnp.einsum("fai,ga->fgi", x, dN[..., 0]), jnp.einsum("fai,ga->fgi", x, dN[..., 1])
    return jnp.einsum("g,fg->", jnp.asarray(face.weights), jnp.linalg.norm(jnp.cross(a1, a2), axis=-1))
