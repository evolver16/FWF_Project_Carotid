"""Export an Abaqus .odb for comparison with fem.abaqus (runs in Abaqus' own Python 2.7, not in this project):

    abaqus python fem/odb_export.py job.odb [more.odb ...]

Writes next to each odb:
    <job>_odb.npz          per frame (all steps): time (total), step, frame; node labels, U, RF (n_frames, n_nodes, 3);
                           element labels, element-mean S (n_frames, n_el, 6: S11 S22 S33 S12 S13 S23) and SDV* means
                           (n_frames, n_el, n_sdv) if written; first part instance only
    <job>_odb_rf.csv       per frame: time and RF1-3 summed over every node set of the assembly and the instance
    <job>_odb_history.csv  all history outputs: region, output, time, value
"""
from __future__ import print_function

import csv
import os
import sys

import numpy as np
from odbAccess import openOdb


def block_arrays(field, labels, attr):
    """field values of one instance -> (n_labels, n_comp) means per label (nodes or elements)"""
    index = dict((l, i) for i, l in enumerate(labels))
    out, count = None, np.zeros(len(labels))
    for b in field.bulkDataBlocks:
        data = np.asarray(b.data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if out is None:
            out = np.zeros((len(labels), data.shape[1]))
        for lab, d in zip(np.asarray(getattr(b, attr)).ravel(), data):
            k = index.get(int(lab))
            if k is not None:
                out[k] += d
                count[k] += 1
    if out is None:
        return None
    return out / np.maximum(count, 1)[:, None]


def export(path):
    odb = openOdb(path, readOnly=True)
    stem = os.path.splitext(path)[0]
    inst = odb.rootAssembly.instances.values()[0]
    nodes = [n.label for n in inst.nodes]
    elements = [e.label for e in inst.elements]
    sets = [("assembly." + k, s) for k, s in odb.rootAssembly.nodeSets.items()]
    sets += [(k, s) for k, s in inst.nodeSets.items()]
    times, steps, frames, U, RF, S, SDV, rf_rows = [], [], [], [], [], [], [], []
    sdv_names = None
    for si, step in enumerate(odb.steps.values()):
        for fi, frame in enumerate(step.frames):
            t = step.totalTime + frame.frameValue
            fo = frame.fieldOutputs
            times.append(t)
            steps.append(si + 1)
            frames.append(fi)
            sub = lambda name: fo[name].getSubset(region=inst) if name in fo.keys() else None
            u, rf = sub("U"), sub("RF")
            U.append(block_arrays(u, nodes, "nodeLabels") if u else np.full((len(nodes), 3), np.nan))
            RF.append(block_arrays(rf, nodes, "nodeLabels") if rf else np.full((len(nodes), 3), np.nan))
            s = sub("S")
            S.append(block_arrays(s, elements, "elementLabels")[:, :6] if s else np.full((len(elements), 6), np.nan))
            names = sorted([k for k in fo.keys() if k.startswith("SDV")], key=lambda k: int(k[3:]) if k[3:].isdigit() else 0)
            if names:
                sdv_names = names
                SDV.append(np.hstack([block_arrays(sub(k), elements, "elementLabels") for k in names]))
            row = {"step": si + 1, "frame": fi, "time": t}
            if rf:
                for name, ns in sets:
                    vals = fo["RF"].getSubset(region=ns).values
                    tot = np.sum([v.data for v in vals], axis=0) if len(vals) else np.zeros(3)
                    for c in range(3):
                        row["RF%d_%s" % (c + 1, name)] = float(tot[c])
            rf_rows.append(row)
        print("%s: step %s, %d frames" % (os.path.basename(path), step.name, len(step.frames)))
    arrays = dict(time=np.array(times), step=np.array(steps), frame=np.array(frames), node_labels=np.array(nodes),
                  U=np.array(U), RF=np.array(RF), element_labels=np.array(elements), S=np.array(S))
    if SDV:
        arrays["SDV"] = np.array(SDV)
        arrays["sdv_names"] = np.array(sdv_names)
    np.savez(stem + "_odb.npz", **arrays)
    keys = sorted(set(k for r in rf_rows for k in r), key=lambda k: (k not in ("step", "frame", "time"), k))
    with open(stem + "_odb_rf.csv", "wb" if sys.version_info[0] < 3 else "w") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rf_rows)
    with open(stem + "_odb_history.csv", "wb" if sys.version_info[0] < 3 else "w") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "region", "output", "time", "value"])
        for step in odb.steps.values():
            for rname, region in step.historyRegions.items():
                for oname, ho in region.historyOutputs.items():
                    for t, v in ho.data:
                        w.writerow([step.name, rname, oname, step.totalTime + t, v])
    odb.close()
    print("written: %s_odb.npz, _odb_rf.csv, _odb_history.csv" % stem)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        export(p)
