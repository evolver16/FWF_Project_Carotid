"""Abaqus UMAT and UHYPER (Fortran) as models of fem.System: compiled with gfortran into a shared library with a bind(C)
driver per batch of integration points and an XLA FFI handler (C++, g++), called as a custom call inside the compiled
JAX program (no Python per call); kinematics (DROT, log strains, Voigt) in jax.numpy. Calls are serialized (legacy
UMATs keep SAVE / COMMON state); the user code works on copies of the inputs.

UMAT:

Increments as in Abaqus/Standard with NLGEOM: every Newton iteration calls the UMAT from the state at the start of the
increment (STRESS, STATEV, DFGRD0, TIME); System.commit closes the increment (state <- end values, TIME += DTIME,
KINC += 1). DDSDDE is taken as the Abaqus Jacobian C of the Jaumann rate of Kirchhoff stress over J, which gives the
consistent Cauchy stress rate for fem.System's forward-mode tangent:

    sigma_dot = C : D + W sigma - sigma W - sigma tr D,   D, W = sym, skew of (dF F^-1)   (Voigt 11,22,33,12,13,23,
                                                                                            engineering shear in D)
STRESS and STRAN on entry are rotated by DROT (rotation of the polar decomposition of DFGRD1 DFGRD0^-1); STRAN, DSTRAN
are logarithmic strains. Supported: PROPS, STATEV, PREDEF / DPRED per point, CMNAME, COORDS, NOEL, NPT, CELENT,
KSTEP (also as JSTEP(4) = step, static, NLGEOM, no perturbation), KINC, TIME, DTIME, SDVINI, XIT (raises).
Not supported: other Abaqus utility routines (SINV, SPRINC, ROTSIG, ...; the build fails with an undefined symbol),
*ORIENTATION (stresses in global axes), temperature, PNEWDT < 1 (ignored), adjoints with respect to the state. Points with a non-finite F or det F <= 0 (trial
iterates of a diverging Newton step) return NaN without calling the UMAT, so the solver's cutbacks take over.

UHYPER (*Hyperelastic, user): see Uhyper.
"""

import ctypes
import hashlib
import pathlib
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from fem.pytree import pytree
from fem.tensor3 import inv3

CACHE = pathlib.Path.home() / ".cache" / "fem_umat"
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

ABA_PARAM = """      IMPLICIT REAL*8(A-H,O-Z)
      PARAMETER (NPRECD=2)
"""

XIT = """
module umat_flags
  integer :: xit_called = 0
end module

subroutine xit()
  use umat_flags
  write(*, *) 'XIT called by the user subroutine'
  xit_called = 1
end subroutine
"""

UMAT_DRIVER = """
subroutine umat_batch(n, nstatv, nprops, npred, stress, statev, ddsdde, sse, stran, dstran, time, dtime, predef, &
                      dpred, props, coords, drot, pnewdt, celent, dfgrd0, dfgrd1, noel, npt, kstep, kinc, cname, &
                      ierr) bind(C, name='umat_batch')
  use iso_c_binding
  use umat_flags
  implicit none
  integer(c_int), intent(in) :: n, nstatv, nprops, npred
  real(c_double) :: stress(6, n), statev(max(nstatv, 1), n), ddsdde(6, 6, n), sse(n), stran(6, n), dstran(6, n)
  real(c_double) :: time(2, n), dtime(n), predef(max(npred, 1), n), dpred(max(npred, 1), n)
  real(c_double) :: props(max(nprops, 1), n), coords(3, n), drot(3, 3, n), pnewdt(n), celent(n)
  real(c_double) :: dfgrd0(3, 3, n), dfgrd1(3, 3, n)
  integer(c_int) :: noel(n), npt(n), kstep(n), kinc(n), ierr
  character(kind=c_char) :: cname(80)
  character(len=80) :: cmname
  real(8) :: spd, scd, rpl, ddsddt(6), drplde(6), drpldt, temp, dtemp
  integer :: i, ndi, nshr, ntens, nst, npr, nel, ipt, layer, kspt, kst(4), kin
  external umat
  do i = 1, 80
    cmname(i:i) = cname(i)
  end do
  ierr = 0
  xit_called = 0
  do i = 1, n
    ndi = 3; nshr = 3; ntens = 6; nst = nstatv; npr = nprops; nel = noel(i); ipt = npt(i)
    layer = 1; kspt = 1; kst = (/ kstep(i), 1, 1, 0 /); kin = kinc(i)
    spd = 0d0; scd = 0d0; rpl = 0d0; ddsddt = 0d0; drplde = 0d0; drpldt = 0d0; temp = 0d0; dtemp = 0d0
    call umat(stress(:, i), statev(:, i), ddsdde(:, :, i), sse(i), spd, scd, rpl, ddsddt, drplde, drpldt, &
              stran(:, i), dstran(:, i), time(:, i), dtime(i), temp, dtemp, predef(:, i), dpred(:, i), cmname, &
              ndi, nshr, ntens, nst, props(:, i), npr, coords(:, i), drot(:, :, i), pnewdt(i), celent(i), &
              dfgrd0(:, :, i), dfgrd1(:, :, i), nel, ipt, layer, kspt, kst, kin)
    if (xit_called /= 0) then
      ierr = i
      return
    end if
  end do
end subroutine
"""

UHYPER_DRIVER = """
subroutine uhyper_batch(n, nprops, nstatv, nfield, bi1, bi2, aj, u, ui1, ui2, ui3, props, statev, fieldv, fieldvinc, &
                        noel, cname, ierr) bind(C, name='uhyper_batch')
  use iso_c_binding
  use umat_flags
  implicit none
  integer(c_int), intent(in) :: n, nprops, nstatv, nfield
  real(c_double) :: bi1(n), bi2(n), aj(n), u(n), ui1(3, n), ui2(6, n), ui3(6, n), props(max(nprops, 1), n)
  real(c_double) :: statev(max(nstatv, 1), n), fieldv(max(nfield, 1), n), fieldvinc(max(nfield, 1), n)
  integer(c_int) :: noel(n), ierr
  character(kind=c_char) :: cname(80)
  character(len=80) :: cmname
  real(8) :: temp
  integer :: i, nel, incmp, nst, nfv, npr
  external uhyper
  do i = 1, 80
    cmname(i:i) = cname(i)
  end do
  ierr = 0
  xit_called = 0
  do i = 1, n
    nel = noel(i); incmp = 0; nst = nstatv; nfv = nfield; npr = nprops; temp = 0d0
    call uhyper(bi1(i), bi2(i), aj(i), u(i), ui1(:, i), ui2(:, i), ui3(:, i), temp, nel, cmname, incmp, nst, &
                statev(:, i), nfv, fieldv(:, i), fieldvinc(:, i), npr, props(:, i))
    if (xit_called /= 0) then
      ierr = i
      return
    end if
  end do
end subroutine
"""

SDVINI_DRIVER = """
subroutine sdvini_batch(n, nstatv, statev, coords, noel, npt) bind(C, name='sdvini_batch')
  use iso_c_binding
  implicit none
  integer(c_int), intent(in) :: n, nstatv
  real(c_double) :: statev(nstatv, n), coords(3, n)
  integer(c_int) :: noel(n), npt(n)
  integer :: i, nst, ncrds, nel, ipt, layer, kspt
  external sdvini
  do i = 1, n
    nst = nstatv; ncrds = 3; nel = noel(i); ipt = npt(i); layer = 1; kspt = 1
    call sdvini(statev(:, i), coords(:, i), nst, ncrds, nel, ipt, layer, kspt)
  end do
end subroutine
"""

FFI_HEAD = r"""
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
using F64 = ffi::Buffer<ffi::F64>;
using S32 = ffi::Buffer<ffi::S32>;
using Out = ffi::ResultBuffer<ffi::F64>;
static std::mutex mu;

template <typename B> static auto copy(const B& b, size_t n) {
  std::vector<std::remove_pointer_t<decltype(b.typed_data())>> v(b.typed_data(), b.typed_data() + b.element_count());
  v.resize(std::max(v.size(), n));
  return v;
}

static std::string name80(std::string_view s) {
  std::string c(s.substr(0, 80));
  c.resize(80, ' ');
  return c;
}
"""

UMAT_FFI = FFI_HEAD + r"""
extern "C" void umat_batch(int*, int*, int*, int*, double*, double*, double*, double*, double*, double*, double*,
                           double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
                           int*, int*, int*, int*, char*, int*);

static ffi::Error umat_impl(F64 stress0, F64 statev0, F64 stran, F64 dstran, F64 time, F64 dtime, F64 predef,
                            F64 dpred, F64 props, F64 coords, F64 drot, F64 celent, F64 dfgrd0, F64 dfgrd1, S32 noel,
                            S32 npt, S32 kstep, S32 kinc, Out stress, Out statev, Out ddsdde,
                            std::string_view cmname) {
  int n = dtime.element_count();
  if (n == 0) return ffi::Error::Success();
  int nstatv = statev0.element_count() / n, nprops = props.element_count() / n, npred = predef.element_count() / n;
  int ierr = 0;
  auto s = copy(stress0, n), sv = copy(statev0, n), e = copy(stran, n), de = copy(dstran, n), t = copy(time, n);
  auto dt = copy(dtime, n), p = copy(predef, n), dp = copy(dpred, n), pr = copy(props, n), x = copy(coords, n);
  auto r = copy(drot, n), c = copy(celent, n), f0 = copy(dfgrd0, n), f1 = copy(dfgrd1, n);
  auto el = copy(noel, n), ip = copy(npt, n), ks = copy(kstep, n), ki = copy(kinc, n);
  std::vector<double> sse(n), pnewdt(n, 1.0);
  std::string name = name80(cmname);
  std::fill_n(ddsdde->typed_data(), ddsdde->element_count(), 0.0);
  {
    std::lock_guard<std::mutex> lock(mu);
    umat_batch(&n, &nstatv, &nprops, &npred, s.data(), sv.data(), ddsdde->typed_data(), sse.data(), e.data(),
               de.data(), t.data(), dt.data(), p.data(), dp.data(), pr.data(), x.data(), r.data(), pnewdt.data(),
               c.data(), f0.data(), f1.data(), el.data(), ip.data(), ks.data(), ki.data(), name.data(), &ierr);
  }
  if (ierr) return ffi::Error::Internal("UMAT called XIT at point " + std::to_string(ierr) + " of the batch");
  std::copy_n(s.data(), stress->element_count(), stress->typed_data());
  std::copy_n(sv.data(), statev->element_count(), statev->typed_data());
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(umat_ffi, umat_impl, ffi::Ffi::Bind()
    .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>()
    .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<S32>().Arg<S32>().Arg<S32>().Arg<S32>()
    .Ret<F64>().Ret<F64>().Ret<F64>().Attr<std::string_view>("cmname"));
"""

UHYPER_FFI = FFI_HEAD + r"""
extern "C" void uhyper_batch(int*, int*, int*, int*, double*, double*, double*, double*, double*, double*, double*,
                             double*, double*, double*, double*, int*, char*, int*);

static ffi::Error uhyper_impl(F64 inv, F64 props, F64 statev, F64 fieldv, S32 noel, Out u, Out ui1, Out ui2,
                              std::string_view cmname) {
  int n = noel.element_count();
  if (n == 0) return ffi::Error::Success();
  int nprops = props.element_count() / n, nstatv = statev.element_count() / n, nfield = fieldv.element_count() / n;
  int ierr = 0;
  std::vector<double> bi1(n), bi2(n), aj(n), ui3(6 * size_t(n)), dfield(std::max(nfield, 1) * size_t(n));
  for (int i = 0; i < n; ++i) {
    bi1[i] = inv.typed_data()[3 * i];
    bi2[i] = inv.typed_data()[3 * i + 1];
    aj[i] = inv.typed_data()[3 * i + 2];
  }
  auto pr = copy(props, n), sv = copy(statev, n), fv = copy(fieldv, n);
  auto el = copy(noel, n);
  std::string name = name80(cmname);
  for (auto o : {&u, &ui1, &ui2}) std::fill_n((*o)->typed_data(), (*o)->element_count(), 0.0);
  {
    std::lock_guard<std::mutex> lock(mu);
    uhyper_batch(&n, &nprops, &nstatv, &nfield, bi1.data(), bi2.data(), aj.data(), u->typed_data(),
                 ui1->typed_data(), ui2->typed_data(), ui3.data(), pr.data(), sv.data(), fv.data(), dfield.data(),
                 el.data(), name.data(), &ierr);
  }
  if (ierr) return ffi::Error::Internal("UHYPER called XIT at point " + std::to_string(ierr) + " of the batch");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(uhyper_ffi, uhyper_impl, ffi::Ffi::Bind()
    .Arg<F64>().Arg<F64>().Arg<F64>().Arg<F64>().Arg<S32>().Ret<F64>().Ret<F64>().Ret<F64>()
    .Attr<std::string_view>("cmname"));
"""


def build(sources, kind="umat", free_form=None, sdvini=False, flags=("-std=legacy",)):
    """Shared library of the user sources + driver of kind "umat" or "uhyper", cached by content (sources, every file
    next to them, flags);
    free_form None: free form if a source contains '!dec$ freeform', else fixed form; flags: gfortran options for the
    user sources, default -std=legacy (argument size / type mismatches between routines are warnings, as with the
    Intel compiler of Abaqus)"""
    sources = [pathlib.Path(s).resolve() for s in sources]
    dirs = sorted({s.parent for s in sources})
    if free_form is None:
        free_form = any("!dec$ freeform" in s.read_text(errors="ignore").lower() for s in sources)
    form = ["-ffree-form", "-ffree-line-length-none"] if free_form else ["-ffixed-form", "-ffixed-line-length-132"]
    driver = XIT + {"umat": UMAT_DRIVER, "uhyper": UHYPER_DRIVER}[kind] + (SDVINI_DRIVER if sdvini else "")
    handler = {"umat": UMAT_FFI, "uhyper": UHYPER_FFI}[kind]
    h = hashlib.sha256(repr((form, list(flags), driver, handler, jax.__version__, ABA_PARAM,
                             [str(s) for s in sources])).encode())
    for d in dirs:
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in (".for", ".f", ".f90", ".inc"):
                h.update(f.name.encode() + f.read_bytes())
    lib = CACHE / h.hexdigest()[:16] / "libumat.so"
    if lib.exists():
        return lib
    lib.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for name in ("aba_param.inc", "ABA_PARAM.INC"):
            (tmp / name).write_text(ABA_PARAM)
        (tmp / "driver.f90").write_text(driver)
        inc = [f"-I{tmp}"] + [f"-I{d}" for d in dirs]
        objs = []
        for i, s in enumerate(sources):
            objs.append(tmp / f"user{i}.o")
            run(["gfortran", "-c", "-O2", "-fPIC", *form, *inc, *flags,
                 str(s), "-o", str(objs[-1])], cwd=tmp)
        (tmp / "handler.cc").write_text(handler)
        run(["gfortran", "-c", "-O2", "-fPIC", str(tmp / "driver.f90"), "-o", str(tmp / "driver.o")], cwd=tmp)
        run(["g++", "-c", "-O2", "-fPIC", "-std=c++17", f"-I{jax.ffi.include_dir()}", str(tmp / "handler.cc"), "-o",
             str(tmp / "handler.o")], cwd=tmp)
        run(["gfortran", "-shared", "-Wl,--no-undefined", *map(str, objs), str(tmp / "driver.o"),
             str(tmp / "handler.o"), "-lstdc++", "-o", str(lib)], cwd=tmp)
    return lib


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"UMAT build failed: {' '.join(cmd)}\n{p.stdout}{p.stderr}")


@pytree(("stress", "statev", "F", "props", "predef", "dpred", "coords", "time", "dtime", "kstep", "kinc", "noel",
         "npt", "celent"))
class UmatState:
    """Start-of-increment values at one integration point (batched: leading (n_el, n_gp))"""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def log_strain(F):
    """Logarithmic strain 1/2 ln(F F^T), (..., 3, 3)"""
    lam, n = jnp.linalg.eigh(F @ T(F))
    return n * (0.5 * jnp.log(lam))[..., None, :] @ T(n)


def T(A):
    return jnp.swapaxes(A, -1, -2)


def to_voigt(A, shear=1.0):
    return jnp.stack([A[..., i, j] * (shear if i != j else 1.0) for i, j in VOIGT], -1)


def from_voigt(v):
    """(..., 6) -> symmetric (..., 3, 3); also the UI2 order of UHYPER -> Hessian"""
    return v[..., jnp.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])]


def nan_where(ok, *arrays):
    return tuple(jnp.where(ok.reshape(ok.shape + (1,) * (a.ndim - ok.ndim)), a, jnp.nan) for a in arrays)


def register(lib, kind):
    """XLA FFI target of a built library, named after its cache key"""
    name = f"{kind}_{lib.parent.name}"
    jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(ctypes.CDLL(str(lib)), f"{kind}_ffi")),
                                platform="cpu")
    return name


class Umat:
    """fem.System model for an Abaqus UMAT; sources: Fortran files (the first defines SUBROUTINE UMAT, others may be
    INCLUDEd from the same folder), nstatv state variables, cmname material name, npred predefined fields,
    sdvini: initial state variables from the user's SUBROUTINE SDVINI"""

    def __init__(self, sources, nstatv=0, cmname="", npred=1, sdvini=False, free_form=None, flags=("-std=legacy",)):
        lib = build(sources, "umat", free_form, sdvini, flags)
        self.lib, self.target = ctypes.CDLL(str(lib)), register(lib, "umat")
        self.nstatv, self.npred, self.sdvini, self.cmname = nstatv, npred, sdvini, cmname
        self._call = jax.custom_jvp(self._primal)
        self._call.defjvp(self._jvp)

    def state(self, mesh, props, predef=None, statev=None, dtime=1.0, kstep=1, time=0.0):
        """Batched initial state for a mesh (2x2x2 / tet10 Gauss points): props (n_props,) or per point
        (n_el, n_gp, n_props), predef likewise (n_pred), statev given / from SDVINI / zero; NOEL, NPT 1-based"""
        from fem.core import gauss_points, geometry
        xg = np.asarray(gauss_points(mesh.X, mesh.conn))
        _, wdet = geometry(mesh.X, mesh.conn)
        n_el, n_gp = xg.shape[:2]
        per = lambda v, k: np.broadcast_to(np.asarray(v, dtype=float), (n_el, n_gp, k)).copy()
        props = np.atleast_1d(np.asarray(props, dtype=float))
        noel = np.broadcast_to(np.arange(1, n_el + 1, dtype=np.int32)[:, None], (n_el, n_gp)).copy()
        npt = np.broadcast_to(np.arange(1, n_gp + 1, dtype=np.int32)[None], (n_el, n_gp)).copy()
        if statev is None:
            statev = np.zeros((n_el, n_gp, self.nstatv))
            if self.sdvini:
                sv = np.zeros((n_el * n_gp, self.nstatv))
                ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
                c = np.ascontiguousarray(xg.reshape(-1, 3))
                self.lib.sdvini_batch(ctypes.byref(ctypes.c_int(len(sv))), ctypes.byref(ctypes.c_int(self.nstatv)),
                                      ptr(sv), ptr(c), ptr(noel.ravel().copy()), ptr(npt.ravel().copy()))
                statev = sv.reshape(n_el, n_gp, self.nstatv)
        return UmatState(
            stress=jnp.zeros((n_el, n_gp, 6)), statev=jnp.asarray(per(statev, self.nstatv)),
            F=jnp.broadcast_to(jnp.eye(3), (n_el, n_gp, 3, 3)), props=jnp.asarray(per(props, props.shape[-1])),
            predef=jnp.asarray(per(np.zeros(self.npred) if predef is None else predef, self.npred)),
            dpred=jnp.zeros((n_el, n_gp, self.npred)), coords=jnp.asarray(xg),
            time=jnp.asarray(per([time, time], 2)), dtime=jnp.full((n_el, n_gp), float(dtime)),
            kstep=jnp.full((n_el, n_gp), kstep, dtype=jnp.int32), kinc=jnp.ones((n_el, n_gp), dtype=jnp.int32),
            noel=jnp.asarray(noel), npt=jnp.asarray(npt), celent=jnp.asarray(np.cbrt(wdet.sum(1))[:, None]
                                                                              * np.ones((1, n_gp))))

    def _primal(self, F, s):
        """UMAT at the end of the increment -> (Cauchy (3,3), STRESS, STATEV, DDSDDE (6,6)); bad points NaN"""
        ok = jnp.isfinite(F).all((-2, -1)) & (jnp.linalg.det(F) > 0) & (jnp.linalg.det(s.F) > 0)
        F1, F0 = jnp.where(ok[..., None, None], F, jnp.eye(3)), jnp.where(ok[..., None, None], s.F, jnp.eye(3))
        dF = F1 @ jnp.linalg.inv(F0)
        lam, v = jnp.linalg.eigh(T(dF) @ dF)
        R = dF @ (v * lam[..., None, :] ** -0.5) @ T(v)
        eps0 = R @ log_strain(F0) @ T(R)
        out = (jax.ShapeDtypeStruct(F.shape[:-2] + (6,), F.dtype),
               jax.ShapeDtypeStruct(F.shape[:-2] + (self.nstatv,), F.dtype),
               jax.ShapeDtypeStruct(F.shape[:-2] + (6, 6), F.dtype))
        i32 = lambda a: a.astype(jnp.int32)
        stress, statev, ddsdde = jax.ffi.ffi_call(self.target, out, vmap_method="broadcast_all")(
            to_voigt(R @ from_voigt(s.stress) @ T(R)), s.statev, to_voigt(eps0, 2.0),
            to_voigt(log_strain(F1) - eps0, 2.0), s.time, s.dtime, s.predef, s.dpred, s.props, s.coords, T(R),
            s.celent, T(F0), T(F1), i32(s.noel), i32(s.npt), i32(s.kstep), i32(s.kinc), cmname=self.cmname)
        return nan_where(ok, from_voigt(stress), stress, statev, T(ddsdde))

    def _jvp(self, primals, tangents):
        F, s = primals
        dF = tangents[0]
        out = self._call(F, s)
        sig, C = out[0], out[3]
        L = dF @ inv3(F)
        D, W = 0.5 * (L + L.T), 0.5 * (L - L.T)
        dsig = from_voigt(C @ to_voigt(D, 2.0)) + W @ sig - sig @ W - sig * jnp.trace(D)
        return out, (dsig, jnp.zeros_like(out[1]), jnp.zeros_like(out[2]), jnp.zeros_like(out[3]))

    def sigma_solver(self, state, F):
        """Cauchy stress at the end of the increment from the start-of-increment state; aux (STRESS, STATEV)"""
        sig, stress, statev, _ = self._call(F, state)
        return sig, (stress, statev)

    def commit(self, state, F, aux):
        """Close the increment: end values become the start of the next one"""
        return state.replace(stress=aux[0], statev=aux[1], F=F, time=state.time + state.dtime, kinc=state.kinc + 1)


@pytree(("props", "statev", "fieldv", "noel"))
class UhyperState:
    """Properties, state and field variables, element number at one integration point (batched: (n_el, n_gp))"""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class Uhyper:
    """fem.System model for an Abaqus UHYPER (*Hyperelastic, user): U(I1bar, I2bar, J) and its derivatives from the
    user subroutine, B = F F^T, Bbar = J^(-2/3) B, I1bar = tr Bbar, I2bar = (I1bar^2 - tr Bbar^2) / 2,

        sigma = (1/J) dU/dF F^T   (AD through the invariants; dU/dI from UI1, d2U/dI2 from UI2 for the tangent)

    UI2 order (I1 I1, I2 I2, J J, I1 I2, I1 J, I2 J) as in Abaqus; UI3 unused. STATEV / FIELDV are passed, not updated.
    J <= 0 or a non-finite F gives NaN (solver cutback)."""

    def __init__(self, sources, nstatv=0, nfield=0, cmname="", free_form=None, flags=("-std=legacy",)):
        self.target = register(build(sources, "uhyper", free_form, False, flags), "uhyper")
        self.nstatv, self.nfield, self.cmname = nstatv, nfield, cmname
        self._U = jax.custom_jvp(lambda inv, s: self._call(inv, s)[0])
        self._U.defjvp(lambda p, t: (self._U(*p), self._dU(*p) @ t[0]))
        self._dU = jax.custom_jvp(lambda inv, s: self._call(inv, s)[1])
        self._dU.defjvp(lambda p, t: (self._dU(*p), self._call(*p)[2] @ t[0]))

    def state(self, mesh, props, statev=None, fieldv=None):
        """Batched state for a mesh: props (n_props,) or per point (n_el, n_gp, n_props)"""
        from fem.core import gauss_points
        n_el, n_gp = np.asarray(gauss_points(mesh.X, mesh.conn)).shape[:2]
        per = lambda v, k: jnp.asarray(np.broadcast_to(np.asarray(v, dtype=float), (n_el, n_gp, k)).copy())
        props = np.atleast_1d(np.asarray(props, dtype=float))
        return UhyperState(props=per(props, props.shape[-1]),
                           statev=per(np.zeros(self.nstatv) if statev is None else statev, self.nstatv),
                           fieldv=per(np.zeros(self.nfield) if fieldv is None else fieldv, self.nfield),
                           noel=jnp.asarray(np.broadcast_to(np.arange(1, n_el + 1, dtype=np.int32)[:, None],
                                                            (n_el, n_gp)).copy()))

    def _call(self, inv, s):
        """UHYPER at invariants (..., 3) -> (U, dU (3), d2U (3, 3)); bad points NaN"""
        ok = jnp.isfinite(inv).all(-1) & (inv[..., 2] > 0)
        out = (jax.ShapeDtypeStruct(inv.shape[:-1], inv.dtype), jax.ShapeDtypeStruct(inv.shape, inv.dtype),
               jax.ShapeDtypeStruct(inv.shape[:-1] + (6,), inv.dtype))
        u, ui1, ui2 = jax.ffi.ffi_call(self.target, out, vmap_method="broadcast_all")(
            jnp.where(ok[..., None], inv, jnp.array([3.0, 3.0, 1.0])), s.props, s.statev, s.fieldv,
            s.noel.astype(jnp.int32), cmname=self.cmname)
        return nan_where(ok, u, ui1, from_voigt(ui2))

    def energy(self, state, F):
        J = jnp.linalg.det(F)
        Bb = J ** (-2.0 / 3.0) * F @ F.T
        I1 = jnp.trace(Bb)
        return self._U(jnp.stack([I1, 0.5 * (I1 ** 2 - jnp.trace(Bb @ Bb)), J]), state)

    def sigma_solver(self, state, F):
        """Cauchy stress (1/J) dU/dF F^T"""
        return jax.grad(lambda f: self.energy(state, f))(F) @ F.T / jnp.linalg.det(F), None

    def commit(self, state, F, aux):
        return state
