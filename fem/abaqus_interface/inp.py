"""Run an Abaqus/Standard .inp with a UMAT or UHYPER (abaqus_interface.umat) or the built-in *Hyperelastic, neo hooke
(materials.NeoHookean, U = C10 (I1bar - 3) + (J - 1)^2 / D1, K = 2 / D1; no user file) on fem.System: one part (one
instance), one material, solid C3D8* / C3D10* / C3D4* elements, *Static steps (NLGEOM) with
    *Boundary        sets / node labels (instance prefix allowed), dof ranges, XSYMM / YSYMM / ZSYMM / ENCASTRE / PINNED
    *Dsload          P (follower pressure) on element-based surfaces
    *Cload           dead nodal forces (sets / node labels, dofs 1-3)
    *Field, *Initial Conditions, type=FIELD     nodal predefined fields -> PREDEF / DPRED at the Gauss points
    *Initial Conditions, type=SOLUTION, user     STATEV from the user's SDVINI
    *Amplitude       TABULAR (linear) or SMOOTH STEP, step time
    *Parameter       name = expression, used as <name>
Without an amplitude a changed value ramps linearly over the step from its value at the start of the step, with an
amplitude it is magnitude x A(step time); values not redefined in a step stay constant; op=NEW removes the previous
definitions of that keyword. PREDEF is the field at the start of the increment, DPRED its increment (as in Abaqus).

Element formulation from the element type: C3D8 -> "fbar" (Abaqus C3D8 integrates the volumetric part selectively
reduced, F-bar is the closest), C3D8H / C3D10H / C3D4H -> "hybrid", C3D10 / C3D4 -> "standard" (same element as
Abaqus), C3D8R -> "fbar". fem's "hybrid"
enforces J = 1 (incompressible); Abaqus' hybrid elements keep a compressible material's K (perturbed Lagrangian), so
C3D8H / C3D10H / C3D4H with a compressible built-in or UHYPER material -> "fbar" (exact for homogeneous deformation).
Increments (Newton starts from the linear extrapolation of the previous two within a step): *Static, direct: fixed initial
increment; otherwise fixed maximum increment (no automatic incrementation;
the solver's load continuation handles non-converging increments). TIME(1) restarts in every step, KINC restarts at 1.
Output (as Abaqus: stress of the converged increment, i.e. from the start-of-increment state, STATEV after the
update): <out>/<job>.pvd (u, stress, SDV element means, every increment), <job>_rf.csv (reaction force sums per
*Boundary set), <job>_fem.npz (layout of odb_export.py: time, node / element labels, U, RF, element-mean S, SDV).

    fem-abaqus job.inp [user.for] [--out dir] [--element fbar]        (or python -m fem.abaqus_interface.inp ...)
    fem-abaqus compare job_fem.npz job_odb.npz job.inp      (after: abaqus python fem/abaqus_interface/odb_export.py job.odb)
    fem-vs-abaqus job.inp [user.for]                        (both, timed: vs_abaqus.py)
"""

import csv
import math
import pathlib
import re

import jax
import numpy as np

import fem
from fem import mesh as meshlib
from fem.elements import element_for
from fem.abaqus_interface.umat import CACHE, Uhyper, Umat
from materials import NeoHookean

if jax.config.jax_compilation_cache_dir is None:
    jax.config.update("jax_compilation_cache_dir", str(CACHE / "xla"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

SYMM = {"XSYMM": (1, 5, 6), "YSYMM": (2, 4, 6), "ZSYMM": (3, 4, 5), "ENCASTRE": (1, 2, 3, 4, 5, 6),
        "PINNED": (1, 2, 3)}
UNSUPPORTED = ("contact pair", "contact", "rigid body", "coupling", "tie", "dynamic", "dload",
               "surface interaction", "orientation", "equation", "mpc", "temperature", "visco", "frequency",
               "buckle", "kinematic coupling", "distributing coupling")


def keywords(text):
    """[(keyword lower, options {lower: value}, data lines)] in file order; *Parameter values substituted for <name>"""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            parts = [p.strip() for p in line[1:].split(",")]
            opts = {p.split("=")[0].strip().lower(): (p.split("=", 1)[1].strip() if "=" in p else True)
                    for p in parts[1:] if p}
            out.append((parts[0].lower(), opts, []))
        elif out:
            out[-1][2].append([t.strip() for t in line.rstrip(",").split(",")])
    env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    for k, _, data in out:
        if k == "parameter":
            for d in data:
                name, expr = ",".join(d).split("=", 1)
                env[name.strip()] = eval(expr.strip(), {"__builtins__": {}}, env)
    sub = lambda t: re.sub(r"<(\w+)>", lambda m: repr(env[m.group(1)]), t)
    return [(k, o, [[sub(t) for t in d] for d in data]) for k, o, data in out if k != "parameter"]


def bare(token):
    """set or node name without the instance prefix ('ELEMENT-1.N1' -> 'n1')"""
    return token.split(".", 1)[1].lower() if "." in token and not token.replace(".", "").lstrip("-").isdigit() \
        else token.lower()


def amplitude(definition):
    """A(t) of an *Amplitude: TABULAR (linear) or SMOOTH STEP, constant outside the given times"""
    opts, data = definition
    v = np.array([float(x) for d in data for x in d if x]).reshape(-1, 2)
    smooth = str(opts.get("definition", "TABULAR")).upper() == "SMOOTH STEP"

    def A(t):
        if t <= v[0, 0]:
            return v[0, 1]
        if t >= v[-1, 0]:
            return v[-1, 1]
        i = np.searchsorted(v[:, 0], t) - 1
        x = (t - v[i, 0]) / (v[i + 1, 0] - v[i, 0])
        x = x ** 3 * (10 - 15 * x + 6 * x ** 2) if smooth else x
        return v[i, 1] + x * (v[i + 1, 1] - v[i, 1])
    return A


class History:
    """Values of one keyword over the steps: key -> (start, magnitude, amplitude); default(key): value of a key not
    defined before (current displacement, zero load, current field)"""

    def __init__(self, default):
        self.now, self.period, self.default = {}, 1.0, default

    def step(self, block, keyword, entries, amps, period):
        """new step: entries(data line, options) -> [(key, magnitude)] of every definition of keyword in the block"""
        self.now = {k: (self.value(k, 1.0, self.period),) * 2 + (None,) for k in self.now}
        self.period = period
        for _, o, data in [b for b in block if b[0] == keyword]:
            if str(o.get("op", "")).upper() == "NEW":
                self.now = {}
            amp = amps[o["amplitude"].lower()] if "amplitude" in o else None
            for d in data:
                for key, mag in entries(d, o):
                    start = self.now[key][1] if key in self.now else self.default(key)
                    self.now[key] = (start, mag, amp)

    def value(self, key, f, t):
        """at step fraction f and step time t"""
        start, mag, amp = self.now[key]
        return mag * amp(t) if amp is not None else start + f * (mag - start)


def node_lookup(kws, m):
    """nodes_of(token): node indices of a node set name or a node label"""
    nsets = {k.lower(): v for k, v in m.nodes.items()}
    label = {int(d[0]): i for i, d in enumerate(d for k, _, data in kws if k == "node" for d in data)}

    def nodes_of(token):
        t = bare(token)
        return np.array([label[int(t)]]) if t.lstrip("-").isdigit() else nsets[t]
    return nodes_of


def bc_entries(d, nodes_of):
    """[(dof, value)] of one *Boundary data line"""
    if len(d) > 1 and d[1].upper() in SYMM:
        dofs, value = [c for c in SYMM[d[1].upper()] if c <= 3], 0.0
    else:
        a = int(d[1])
        b = int(d[2]) if len(d) > 2 and d[2] else a
        dofs, value = [c for c in range(a, b + 1) if c <= 3], float(d[3]) if len(d) > 3 and d[3] else 0.0
    return [(3 * n + c - 1, value) for n in nodes_of(d[0]) for c in dofs]


def steps(kws, nodes_of, amps, bc_default=lambda k: 0.0):
    """per *Step: (keywords of the step, period, number of increments, History of the *Boundary dofs, *Dsload
    surfaces, *Cload dofs); the History objects are updated in place by the next step"""
    bcs, loads, cloads = History(bc_default), History(lambda k: 0.0), History(lambda k: 0.0)
    for s0 in [i for i, (k, _, _) in enumerate(kws) if k == "step"]:
        block = []
        for k, o, data in kws[s0 + 1:]:
            if k == "end step":
                break
            block.append((k, o, data))
        static = next(((o, d) for k, o, d in block if k == "static"), ({}, [["1.0", "1.0"]]))
        vals = [float(v) for v in static[1][0] if v] if static[1] and static[1][0] and static[1][0][0] else [1., 1.]
        period = vals[1] if len(vals) > 1 else 1.0
        dt = vals[0] if "direct" in static[0] else (vals[3] if len(vals) > 3 else period)
        bcs.step(block, "boundary", lambda d, o: bc_entries(d, nodes_of), amps, period)
        loads.step(block, "dsload", lambda d, o: [(bare(d[0]), float(d[2]))], amps, period)
        cloads.step(block, "cload", lambda d, o: [(3 * n + int(d[1]) - 1, float(d[2])) for n in nodes_of(d[0])
                                                  if int(d[1]) <= 3], amps, period)
        yield block, period, int(np.ceil(period / dt - 1e-9)), bcs, loads, cloads


def run_inp(inp, user, out=None, element=None, flags=("-std=legacy",), log=print):
    """Run inp with the user subroutine file(s) user -> (u, states, rows of reaction forces per increment)"""
    inp = pathlib.Path(inp)
    out = pathlib.Path(out) if out else inp.parent / "fem_out"
    out.mkdir(parents=True, exist_ok=True)
    kws = keywords(inp.read_text())
    bad = sorted({k for k, _, _ in kws if k in UNSUPPORTED})
    bad += [f"initial conditions, type={o.get('type')}" for k, o, _ in kws if k == "initial conditions"
            and str(o.get("type", "")).upper() not in ("FIELD", "SOLUTION")]
    if bad or sum(k == "part" for k, _, _ in kws) > 1:
        raise NotImplementedError(f"{inp.name}: not supported: {', '.join('*' + b for b in bad) or 'several parts'}")
    m = meshlib.read_inp(inp)
    surfaces = {k.lower(): v for k, v in m.faces.items()}
    labels = [int(d[0]) for k, _, data in kws if k == "node" for d in data]
    nodes_of = node_lookup(kws, m)
    elabels = [int(d[0]) for k, _, data in kws if k == "element" for d in data if d and d[0]]
    etype = next(str(o.get("type", "")).upper() for k, o, _ in kws if k == "element")
    amps = {o["name"].lower(): amplitude((o, data)) for k, o, data in kws if k == "amplitude"}

    materials = [i for i, (k, _, _) in enumerate(kws) if k == "material"]
    if len(materials) != 1:
        raise NotImplementedError(f"{inp.name}: exactly one *Material supported, found {len(materials)}")
    i0 = materials[0]
    name = kws[i0][1]["name"]
    props, kind, nstatv, incompressible = [], None, 0, False
    for k, o, data in kws[i0 + 1:]:
        if k not in ("hyperelastic", "user material", "depvar", "density", "user defined field"):
            break
        if k == "hyperelastic" and "user" in o:
            kind, incompressible = "uhyper", str(o.get("type", "")).upper() == "INCOMPRESSIBLE"
            props = [float(v) for d in data for v in d if v]
        elif k == "hyperelastic" and "neo hooke" in o:
            kind = "neohooke"
            props = [float(v) for v in data[0][:2]]
        elif k == "user material":
            kind = "umat"
            props = [float(v) for d in data for v in d if v]
        elif k == "depvar":
            nstatv = int(data[0][0])
    if kind is None:
        raise NotImplementedError(f"{inp.name}: material {name} has neither *Hyperelastic, user / neo hooke nor "
                                  f"*User Material")
    if kind == "neohooke" and props[1] <= 0:
        raise NotImplementedError(f"{inp.name}: incompressible *Hyperelastic, neo hooke (D1 = 0)")

    field_vars = sorted({int(o.get("variable", 1)) for k, o, _ in kws if k == "field"
                         or (k == "initial conditions" and str(o.get("type")).upper() == "FIELD")})
    npred = max(field_vars, default=1)
    nodal = np.zeros((npred, m.n_nodes))
    for k, o, data in kws:
        if k == "initial conditions" and str(o.get("type")).upper() == "FIELD":
            for d in data:
                nodal[int(o.get("variable", 1)) - 1, nodes_of(d[0])] = float(d[1])
    sdvini = any(k == "initial conditions" and str(o.get("type")).upper() == "SOLUTION" and "user" in o
                 for k, o, _ in kws)
    el = element_for(m.conn.shape[1])
    N = np.stack([el.N(g) for g in el.gauss])
    gauss = lambda field: field[:, m.conn[:, :1]].transpose(1, 2, 0) + np.einsum(
        "ga,pea->egp", N, field[:, m.conn] - field[:, m.conn[:, :1]])

    compressible = kind == "neohooke" or (kind == "uhyper" and not incompressible)
    element = element or ("hybrid" if (etype.endswith("H") and not compressible) or incompressible else
                          "standard" if etype[:5] in ("C3D10", "C3D4") and not etype.endswith("H") else "fbar")
    users = [user] if isinstance(user, (str, pathlib.Path)) else list(user or [])
    if kind == "umat":
        model = Umat(users, nstatv=nstatv, cmname=name, npred=npred, sdvini=sdvini, flags=flags)
        states = model.state(m, props, predef=gauss(nodal))
    elif kind == "uhyper":
        model = Uhyper(users, cmname=name, flags=flags)
        states = model.state(m, props)
    else:
        model, states = None, fem.broadcast_state(NeoHookean(props[0], 2.0 / props[1]), m.n_elem, len(el.gauss))
    loaded = sorted({bare(d[0]) for k, _, data in kws if k == "dsload" for d in data})
    if any(d[1].upper() != "P" for k, _, data in kws if k == "dsload" for d in data):
        raise NotImplementedError(f"{inp.name}: *Dsload other than P")
    faces = np.concatenate([surfaces[s] for s in loaded]) if loaded else None
    face_of = {s: np.arange(sum(len(surfaces[r]) for r in loaded[:i]), sum(len(surfaces[r]) for r in loaded[:i + 1]))
               for i, s in enumerate(loaded)}
    sysm = fem.System(m, model, pressure_faces=faces, element=element)
    log(f"{inp.name}: {m.n_elem} {etype} ({element}), {m.n_nodes} nodes, material {name} ({kind.upper()}, "
        f"props {np.round(props, 6).tolist()}{f', {nstatv} state variables, {npred} fields' if kind == 'umat' else ''})"
        + (f", pressure on {', '.join(loaded)}" if loaded else ""))

    fields = History(lambda k: nodal[k[0] - 1, k[1]])
    series = meshlib.VTKSeries(out / f"{inp.stem}.pvd")
    u = np.zeros(sysm.n_dof)
    sdv = lambda st: np.asarray(st.statev).mean(1) if kind == "umat" else np.zeros((m.n_elem, 0))
    series.write(0.0, m.X, m.conn, *sysm.fields(u, states))
    rows, t = [], 0.0
    frames = {"time": [0.0], "U": [u.reshape(-1, 3)], "RF": [np.zeros((m.n_nodes, 3))],
              "S": [np.zeros((m.n_elem, 6))], "SDV": [sdv(states)]}
    for kstep, (block, period, n_inc, bcs, loads, cloads) in enumerate(steps(kws, nodes_of, amps, lambda k: u[k]), 1):
        fields.step(block, "field", lambda d, o: [((int(o.get("variable", 1)), n), float(d[1]))
                                                   for n in nodes_of(d[0])], amps, period)
        groups = {}
        for _, o, data in [b for b in block if b[0] == "boundary"]:
            for d in data:
                groups.setdefault(d[0], set()).update(k for k, _ in bc_entries(d, nodes_of))
        dofs = np.array(sorted(bcs.now))
        log(f"  step {kstep}: period {period:g}, {n_inc} increments, {len(dofs)} prescribed dofs")

        def field_at(f, ts):
            nod = nodal.copy()
            for (var, n) in fields.now:
                nod[var - 1, n] = fields.value((var, n), f, ts)
            return nod

        if kind == "umat":
            time = np.asarray(states.time).copy()
            time[..., 0] = 0.0
            states = states.replace(kstep=states.kstep * 0 + kstep, kinc=states.kinc * 0 + 1,
                                    dtime=states.dtime * 0 + period / n_inc, time=time)
        for inc in range(1, n_inc + 1):
            f, ts = inc / n_inc, inc * period / n_inc
            p, f_dead = 0.0, None
            if loaded:
                p = np.zeros(len(faces))
                for s in loaded:
                    p[face_of[s]] = loads.value(s, f, ts) if s in loads.now else 0.0
            if cloads.now:
                f_dead = np.zeros(sysm.n_dof)
                for k in cloads.now:
                    f_dead[k] += cloads.value(k, f, ts)
            bc = fem.BC(dofs, np.array([bcs.value(d, f, ts) for d in dofs]), f_dead=f_dead, p=p)
            if kind == "umat":
                f0 = field_at(f - 1 / n_inc, ts - period / n_inc)
                f1 = field_at(f, ts)
                states = states.replace(predef=gauss(f0), dpred=gauss(f1 - f0))
            guess = u + (u - u_prev) if inc > 1 else u
            u_prev = u
            u = np.asarray(sysm.equilibrium(states, bc, guess))
            rf = sysm.reaction(u, states, bc)
            sig, pre = sysm.total_stress(u, states).mean(1), states
            states = sysm.commit(u, states)
            if kind == "umat":
                nodal = f1
            t += period / n_inc
            row = {"step": kstep, "inc": inc, "time": t, "newton": sysm.last_iterations}
            for g, ds in groups.items():
                ds = np.array(sorted(ds))
                for c in range(3):
                    row[f"RF{c + 1}_{g}"] = float(rf[ds[ds % 3 == c]].sum())
            rows.append(row)
            frames["time"].append(t)
            frames["U"].append(u.reshape(-1, 3))
            frames["RF"].append(rf.reshape(-1, 3))
            frames["S"].append(np.stack([sig[:, i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], -1))
            frames["SDV"].append(sdv(states))
            series.write(t, m.X, m.conn, *sysm.fields(u, pre, {"SDV": np.asarray(states.statev)}
                                                       if kind == "umat" else None))
            if inc == n_inc or inc % max(n_inc // 10, 1) == 0:
                U = u.reshape(-1, 3)
                log(f"    inc {inc:4d}  time {t:9.4f}  Newton {sysm.last_iterations:3d}  max u "
                    f"{np.abs(U).max(0).round(6).tolist()}  " + "  ".join(
                        f"RF{c + 1} {g} {row[f'RF{c + 1}_{g}']:+.4g}" for g in groups for c in range(3)
                        if abs(row[f"RF{c + 1}_{g}"]) > 1e-9))
    with open(out / f"{inp.stem}_rf.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    np.savez(out / f"{inp.stem}_fem.npz", node_labels=np.array(labels), element_labels=np.array(elabels),
             **{k: np.array(v) for k, v in frames.items()})
    log(f"  output: {out / (inp.stem + '.pvd')}, {inp.stem}_rf.csv, {inp.stem}_fem.npz")
    return u, states, rows


def compare(fem_npz, odb_npz, inp=None, log=print):
    """fem.abaqus vs Abaqus (odb_export.py) at the times both have: max |dU| / max |U|, max |dS| / max |S| of the
    element means, SDV, reaction force sums over the *Boundary sets of inp (all nodes if inp is None)"""
    a, b = np.load(fem_npz), np.load(odb_npz)
    node = {int(l): i for i, l in enumerate(b["node_labels"])}
    elem = {int(l): i for i, l in enumerate(b["element_labels"])}
    ni = np.array([node[int(l)] for l in a["node_labels"]])
    ei = np.array([elem[int(l)] for l in a["element_labels"]])
    sets = {"all": np.arange(len(ni))}
    if inp is not None:
        m = meshlib.read_inp(inp)
        names = {k.lower(): v for k, v in m.nodes.items()}
        sets = {d[0]: names[bare(d[0])] for k, _, data in keywords(pathlib.Path(inp).read_text()) if k == "boundary"
                for d in data if bare(d[0]) in names}
    log(f"{pathlib.Path(fem_npz).name} vs {pathlib.Path(odb_npz).name}")
    for i, t in enumerate(a["time"]):
        j = np.flatnonzero(np.isclose(b["time"], t, atol=1e-6))
        if not len(j) or t == 0:
            continue
        j = j[-1]
        rel = lambda x, y: np.abs(x - y).max() / max(np.abs(y).max(), 1e-300)
        line = f"  time {t:9.4f}: U {rel(a['U'][i], b['U'][j][ni]):.2e}, S {rel(a['S'][i], b['S'][j][ei]):.2e}"
        if "SDV" in b.files and a["SDV"].shape[-1]:
            k = min(a["SDV"].shape[-1], b["SDV"].shape[-1])
            line += f", SDV {rel(a['SDV'][i][:, :k], b['SDV'][j][ei][:, :k]):.2e}"
        log(line + " (relative max)  " + "  ".join(
            f"RF {g} {a['RF'][i][ns].sum(0).round(5)} vs {b['RF'][j][ni[ns]].sum(0).round(5)}" for g, ns in sets.items()))


def main(argv=None):
    """fem-abaqus job.inp user.for [...] [--out dir] [--element fbar];  fem-abaqus compare job_fem.npz job_odb.npz [job.inp]"""
    import argparse
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "compare":
        compare(*argv[1:4])
        return
    p = argparse.ArgumentParser(prog="fem-abaqus", description="Run an Abaqus .inp with a UMAT / UHYPER on fem.System")
    p.add_argument("inp")
    p.add_argument("user", nargs="*")
    p.add_argument("--out")
    p.add_argument("--element")
    a = p.parse_args(argv)
    run_inp(a.inp, a.user, a.out, a.element)


if __name__ == "__main__":
    main()
