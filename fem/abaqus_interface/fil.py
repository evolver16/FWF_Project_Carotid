"""Read an Abaqus binary results file (.fil, written by *Node file / *El file) without Abaqus: Fortran sequential
records of 4096-byte blocks, inside a stream of 8-byte words; every record is (length n, key, n - 2 data words),
reals as float64, integers as int32 in the lower half of a word (the upper half is not always zero)
(Abaqus Analysis User's Manual, "Results file output format").

Keys read: 2000 increment start (total time, step time, ..., step, increment), 1 element header (element, integration
point), 101 U (node, U1-U3), 5 SDV, 30 DG (F11 F22 F33 F12 F13 F23 F21 F31 F32), 11 S; other keys skipped.
"""

import pathlib

import numpy as np


def words(path):
    """the 8-byte word stream (block markers removed)"""
    raw = pathlib.Path(path).read_bytes()
    out, i = [], 0
    while i + 4 <= len(raw):
        n = int(np.frombuffer(raw[i:i + 4], np.int32)[0])
        out.append(raw[i + 4:i + 4 + n])
        i += n + 8
    return b"".join(out)


def read_fil(path):
    """-> {"time", "step", "inc": (n_frames,), "U": {node: (n_frames, 3)}, "SDV" / "DG" / "S": {(el, ip): (n_frames,
    k)}} per increment written"""
    w = words(path)
    ints, reals = np.frombuffer(w, np.int32)[0::2], np.frombuffer(w, np.float64)
    frames, el = [], None
    i = 0
    while i < len(ints):
        n, key = int(ints[i]), int(ints[i + 1])
        if n < 2:
            break
        if key == 2000:
            frames.append(dict(time=reals[i + 2], step=int(ints[i + 7]), inc=int(ints[i + 8]), U={}, SDV={}, DG={},
                               S={}))
        elif key == 1:
            el = (int(ints[i + 2]), int(ints[i + 3]))
        elif key == 101 and frames:
            frames[-1]["U"][int(ints[i + 2])] = reals[i + 3:i + n].copy()
        elif key in (5, 30, 11) and frames and el is not None:
            frames[-1][{5: "SDV", 30: "DG", 11: "S"}[key]][el] = reals[i + 2:i + n].copy()
        i += n
    out = {k: np.array([f[k] for f in frames]) for k in ("time", "step", "inc")}
    for k in ("U", "SDV", "DG", "S"):
        keys = sorted({key for f in frames for key in f[k]})
        out[k] = {key: np.array([f[k].get(key, np.full(len(frames[0][k][keys[0]]) if keys else 0, np.nan))
                                 for f in frames]) for key in keys}
    return out
