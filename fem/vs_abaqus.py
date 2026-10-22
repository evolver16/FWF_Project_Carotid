"""Run one .inp with Abaqus and with fem.abaqus one after the other, time both and compare the results.

    fem-vs-abaqus job.inp user.for [more.for] [--abaqus abaqus] [--cpus 1] [--out dir] [--repeat 1] [--grace 20]

Needs Abaqus (command --abaqus) and gfortran on the same machine. Steps, in <out> (default <inp folder>/vs_<job>):
1. abaqus job=<job> input=<inp> user=<first user file> cpus=<n> mp_mode=threads interactive ask_delete=OFF;
   completion is read from <job>.sta; if the driver has not exited <grace> s after "COMPLETED" (Abaqus 2017 on newer
   Linux hangs there), its process group is terminated and this is reported
2. abaqus python fem/odb_export.py <job>.odb -> <job>_odb.npz
3. fem.abaqus.run_inp (repeat times; the first run includes compiling the user subroutine and JAX tracing)
4. fem.abaqus.compare, timing table -> <job>_timing.json
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import time

from fem.abaqus import compare, run_inp

EXPORT = pathlib.Path(__file__).with_name("odb_export.py")


def run_abaqus(inp, user, out, abaqus, cpus, grace, log):
    """-> (wall time until the .sta reports completion, wall time until the driver exited / was stopped, status)"""
    job = inp.stem
    shutil.copy(inp, out / inp.name)
    for f in out.glob(f"{job}.*"):
        if f.suffix.lower() in (".lck", ".sta", ".odb", ".msg", ".dat"):
            f.unlink()
    cmd = [abaqus, f"job={job}", f"input={inp.name}", f"user={user}", f"cpus={cpus}", "mp_mode=threads",
           "interactive", "ask_delete=OFF"]
    log("abaqus: " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=out, stdout=open(out / f"{job}_abaqus.log", "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    done, status = None, None
    sta = out / f"{job}.sta"
    while proc.poll() is None:
        text = sta.read_text(errors="ignore") if sta.exists() else ""
        if done is None and ("COMPLETED SUCCESSFULLY" in text or "HAS NOT BEEN COMPLETED" in text):
            done = time.time() - t0
            status = "completed" if "COMPLETED SUCCESSFULLY" in text else "not completed"
        if done is not None and time.time() - t0 - done > grace:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(10)
            status += f", driver hung after completion, stopped after {grace:g} s"
            break
        time.sleep(0.2)
    total = time.time() - t0
    if done is None:
        text = sta.read_text(errors="ignore") if sta.exists() else ""
        status = "completed" if "COMPLETED SUCCESSFULLY" in text else f"failed (exit {proc.returncode})"
        done = total
    msg = (out / f"{job}.msg").read_text(errors="ignore") if (out / f"{job}.msg").exists() else ""
    wall = re.findall(r"WALLCLOCK TIME \(SEC\)\s*=\s*([\d.]+)", msg)
    cpu = re.findall(r"TOTAL CPU TIME \(SEC\)\s*=\s*([\d.]+)", msg)
    return dict(until_completed_s=done, until_exit_s=total, status=status,
                solver_wallclock_s=float(wall[-1]) if wall else None, solver_cpu_s=float(cpu[-1]) if cpu else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fem-vs-abaqus", description=__doc__.split("\n\n")[0])
    p.add_argument("inp")
    p.add_argument("user", nargs="+")
    p.add_argument("--abaqus", default="abaqus")
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--grace", type=float, default=20.0)
    p.add_argument("--skip-abaqus", action="store_true", help="reuse <job>_odb.npz from an earlier run")
    a = p.parse_args(argv)
    inp = pathlib.Path(a.inp).resolve()
    users = [pathlib.Path(u).resolve() for u in a.user]
    out = pathlib.Path(a.out).resolve() if a.out else inp.parent / f"vs_{inp.stem}"
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    timing = {"job": inp.stem, "abaqus": None, "odb_export_s": None, "fem_s": []}
    if not a.skip_abaqus:
        timing["abaqus"] = run_abaqus(inp, users[0], out, a.abaqus, a.cpus, a.grace, log)
        log(f"abaqus: {timing['abaqus']}")
        t0 = time.time()
        subprocess.run([a.abaqus, "python", str(EXPORT), f"{inp.stem}.odb"], cwd=out, check=True)
        timing["odb_export_s"] = time.time() - t0
    for r in range(a.repeat):
        t0 = time.time()
        run_inp(out / inp.name if (out / inp.name).exists() else inp, users, out / "fem", log=log)
        timing["fem_s"].append(time.time() - t0)
        log(f"fem run {r + 1}: {timing['fem_s'][-1]:.1f} s")
    odb = out / f"{inp.stem}_odb.npz"
    if odb.exists():
        compare(out / "fem" / f"{inp.stem}_fem.npz", odb, inp, log=log)
    ab = timing["abaqus"] or {}
    log(f"\ntiming {inp.stem}:")
    if ab:
        log(f"  Abaqus  until completed {ab['until_completed_s']:.1f} s (solver wallclock {ab['solver_wallclock_s']} s, "
            f"CPU {ab['solver_cpu_s']} s), until exit {ab['until_exit_s']:.1f} s  [{ab['status']}]")
        log(f"  odb export {timing['odb_export_s']:.1f} s")
    for r, t in enumerate(timing["fem_s"]):
        log(f"  fem run {r + 1}: {t:.1f} s" + (" (incl. gfortran build if not cached, JAX tracing)" if r == 0 else ""))
    (out / f"{inp.stem}_timing.json").write_text(json.dumps(timing, indent=2))
    (out / f"{inp.stem}_vs_abaqus.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
