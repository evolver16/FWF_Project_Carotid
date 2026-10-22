"""Run one .inp with Abaqus and with fem.System (abaqus_interface.inp) one after the other, time both and compare.

    fem-vs-abaqus job.inp [user.for more.for] [--abaqus abaqus] [--cpus 1] [--out dir] [--repeat 1] [--grace 20]

Split between the host (Abaqus, no JAX needed: only the standard library is imported) and the dev container:
    host:       python3 fem/abaqus_interface/vs_abaqus.py job.inp user.for --abaqus-only   (--export-only: odb only)
    container:  fem-vs-abaqus job.inp user.for --skip-abaqus      (same <out>, reads the Abaqus timing and odb export)

Steps, in <out> (default <inp folder>/vs_<job>):
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

EXPORT = pathlib.Path(__file__).with_name("odb_export.py")
ABA_PARAM = "      IMPLICIT REAL*8(A-H,O-Z)\n      PARAMETER (NPRECD=2)\n"
ENV = """# written by fem-vs-abaqus: free-form user subroutine (!dec$ freeform) compiled with gfortran
try:
    _cmd = compile_fortran if isinstance(compile_fortran, (list, tuple)) else [compile_fortran]
    if 'gfortran' in ' '.join(_cmd):
        compile_fortran = list(_cmd) + ['-ffree-form', '-ffree-line-length-none', '-std=legacy']
except NameError:
    pass
"""


def prepare(user, out, log):
    """Linux file names are case sensitive (the UMATs may come from Windows): an INCLUDE of aba_param.inc in another
    case gets that file next to the sources; free-form sources get a local abaqus_v6.env with gfortran free-form flags"""
    files = [f for f in user.parent.iterdir() if f.suffix.lower() in (".for", ".f", ".f90")]
    free = False
    for f in files:
        text = f.read_text(errors="ignore")
        free |= "!dec$ freeform" in text.lower()
        for name in set(re.findall(r"include\s+'([^']+)'", text, flags=re.I)):
            if name.lower() == "aba_param.inc" and name != "aba_param.inc" and not (user.parent / name).exists():
                (user.parent / name).write_text(ABA_PARAM)
                log(f"created {user.parent / name} (Abaqus only provides aba_param.inc)")
    if free:
        (out / "abaqus_v6.env").write_text(ENV)
        log(f"free-form sources: {out / 'abaqus_v6.env'} adds gfortran free-form flags (no effect with Intel Fortran)")


def run_abaqus(inp, user, out, abaqus, cpus, grace, log):
    """-> (wall time until the .sta reports completion, wall time until the driver exited / was stopped, status)"""
    job = inp.stem
    shutil.copy(inp, out / inp.name)
    if user:
        prepare(user, out, log)
    for f in out.glob(f"{job}.*"):
        if f.suffix.lower() in (".lck", ".sta", ".odb", ".msg", ".dat"):
            f.unlink()
    cmd = [abaqus, f"job={job}", f"input={inp.name}", *([f"user={user}"] if user else []), f"cpus={cpus}",
           "mp_mode=threads", "interactive", "ask_delete=OFF"]
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
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            status += f", driver hung after completion, stopped after {grace:g} s"
            break
        time.sleep(0.2)
    total = time.time() - t0
    if done is None:
        text = sta.read_text(errors="ignore") if sta.exists() else ""
        status = "completed" if "COMPLETED SUCCESSFULLY" in text else (
            f"failed (exit {proc.returncode}), see {out / (job + '_abaqus.log')} and {job}.dat / {job}.msg")
        done = total
    return dict(until_completed_s=done, until_exit_s=total, status=status, **solver_times(out / f"{job}.msg"))


def solver_times(msg):
    """WALLCLOCK and TOTAL CPU TIME of the job time summary in the .msg file"""
    text = msg.read_text(errors="ignore") if msg.exists() else ""
    wall = re.findall(r"WALLCLOCK TIME \(SEC\)\s*=\s*([\d.]+)", text)
    cpu = re.findall(r"TOTAL CPU TIME \(SEC\)\s*=\s*([\d.]+)", text)
    return dict(solver_wallclock_s=float(wall[-1]) if wall else None, solver_cpu_s=float(cpu[-1]) if cpu else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fem-vs-abaqus", description=__doc__.split("\n\n")[0])
    p.add_argument("inp")
    p.add_argument("user", nargs="*")
    p.add_argument("--abaqus", default="abaqus")
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--grace", type=float, default=20.0)
    p.add_argument("--skip-abaqus", action="store_true", help="reuse <job>_odb.npz and timing from an earlier run")
    p.add_argument("--abaqus-only", action="store_true", help="only Abaqus and the odb export (host without JAX)")
    p.add_argument("--export-only", action="store_true", help="only the odb export of an existing <out>/<job>.odb")
    a = p.parse_args(argv)
    inp = pathlib.Path(a.inp).resolve()
    users = [pathlib.Path(u).resolve() for u in a.user]
    out = pathlib.Path(a.out).resolve() if a.out else inp.parent / f"vs_{inp.stem}"
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    stamp = out / f"{inp.stem}_timing.json"
    timing = {"job": inp.stem, "abaqus": None, "odb_export_s": None, "fem_s": []}
    if a.skip_abaqus and stamp.exists():
        timing.update({k: v for k, v in json.loads(stamp.read_text()).items() if k != "fem_s"})
    if a.export_only and stamp.exists():
        timing.update({k: v for k, v in json.loads(stamp.read_text()).items() if k != "fem_s"})
    if a.export_only and not timing["abaqus"]:
        timing["abaqus"] = dict(until_completed_s=None, until_exit_s=None, status="from .msg",
                                **solver_times(out / f"{inp.stem}.msg"))
    if not a.skip_abaqus:
        if not a.export_only:
            timing["abaqus"] = run_abaqus(inp, users[0] if users else None, out, a.abaqus, a.cpus, a.grace, log)
            log(f"abaqus: {timing['abaqus']}")
        if not (out / f"{inp.stem}.odb").exists():
            raise SystemExit(f"no {out / (inp.stem + '.odb')}: {(timing['abaqus'] or {}).get('status')}")
        t0 = time.time()
        subprocess.run([a.abaqus, "python", str(EXPORT), f"{inp.stem}.odb"], cwd=out, check=True)
        timing["odb_export_s"] = time.time() - t0
    if a.abaqus_only or a.export_only:
        stamp.write_text(json.dumps(timing, indent=2))
        log(f"Abaqus done: {out}; now in the container: fem-vs-abaqus {a.inp} {' '.join(a.user)} --skip-abaqus"
            + (f" --out {a.out}" if a.out else ""))
        return
    from fem.abaqus_interface.inp import compare, run_inp
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
        log(f"  Abaqus  until completed {ab['until_completed_s'] or float('nan'):.1f} s (solver wallclock {ab['solver_wallclock_s']} s, "
            f"CPU {ab['solver_cpu_s']} s), until exit {ab['until_exit_s'] or float('nan'):.1f} s  [{ab['status']}]")
        log(f"  odb export {timing['odb_export_s']:.1f} s")
    for r, t in enumerate(timing["fem_s"]):
        log(f"  fem run {r + 1}: {t:.1f} s" + (" (incl. gfortran build if not cached, JAX tracing)" if r == 0 else ""))
    stamp.write_text(json.dumps(timing, indent=2))
    (out / f"{inp.stem}_vs_abaqus.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
