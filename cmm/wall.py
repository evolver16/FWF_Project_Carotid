"""Wall composition: constituents as recipes (Part), assembled into HCMM states at every Gauss point."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

import fcmm
import hcmm
import materials
from setups import G_fiber


@dataclass(frozen=True)
class Part:
    """One constituent; its values are par["<name>.<parameter>"] (per unit mass fraction: stress = rho x material stress).
    material "neo-hooke": rho, C10 (+ K unless incompressible, which needs the hybrid element), stress-free state G_e of
    the prestress fixed point; "fung": rho, k1, k2 along direction "theta", "z", "+diag", "-diag" (diagonals at alpha
    from the axis), deposition stretch g.
    laws: None (no turnover), "turnover" (the wall's default laws, assemble(laws=...)), "fiber" (MaesProduction: T, k_plus,
    k_minus, k_tau), "wall" (MixtureProduction + MixtureRemoval: T, k_sigma, k_tau, optional k_turn: turnover rate
    (1 + k_turn dsig^2) / T, Latorre & Humphrey 2018), "removal" (constant removal, T);
    incompressible: None takes the wall's default. lesion: density x f (lesion fraction per Gauss point), else x (1 - f).
    reorient: fiber deposited along the production's M_dep (hcmm.Reorienting, initially M; set per Gauss point).
    growth: None (the mixture's F_g) or "fixed" (hcmm.Fixed: F_g^j = I, stress-free state fixed, e.g. elastin)"""
    name: str
    material: str
    direction: str = ""
    laws: str = None
    incompressible: bool = None
    lesion: bool = False
    reorient: bool = False
    growth: str = None


def parameters(groups, other, kw):
    """{"<constituent>.<parameter>": value} from groups {constituent: {parameter: value}} and other (loads, ...);
    kw: a full name sets one value, a bare parameter name (e.g. k_tau) sets it in every constituent that has it"""
    par = {f"{c}.{k}": v for c, d in groups.items() for k, v in d.items()} | other
    for k, v in kw.items():
        keys = [k] if k in par else [q for q in par if q.split(".")[-1] == k]
        if not keys:
            raise KeyError(f"unknown parameter {k!r}")
        par.update(dict.fromkeys(keys, v))
    return {k: jnp.asarray(v, dtype=float) for k, v in par.items()}


def principal_frame(s_tt, s_zz, s_tz):
    """in-plane principal frame in the wall basis (theta, z): the minor principal direction at beta from e_z towards
    e_theta (mod pi), 2 beta = atan2(-2 s_tz, s_tt - s_zz); -> (beta, sigma_c, sigma_a): sigma_a (minor) along it,
    sigma_c (major) normal to it"""
    d = s_zz - s_tt
    m, r = 0.5 * (s_tt + s_zz), jnp.sqrt(0.25 * d ** 2 + s_tz ** 2)
    return 0.5 * jnp.arctan2(-2 * s_tz, -d), m + r, m - r


@partial(jax.jit, static_argnames=("wall", "mode", "laws", "incompressible", "growth", "n_max"))
def assemble(G_e, Qloc, par, wall, mode="deposition", ds=10.0, lesion=None, laws="fiber", incompressible=False,
             growth="radial", fields=None, n_max=0):
    """HCMM mixture per Gauss point from the parts of wall; Qloc = (e_r, e_theta, e_z) per Gauss point;
    laws, incompressible: defaults for the parts that do not set their own;
    growth: mixture F_g "radial" (hcmm.Anisotropic(e_r), thickening), "axial" (hcmm.Anisotropic(e_z), lengthening) or
    "isotropic" (hcmm.Isotropic); it also moves the stress-free state of every part whose growth is not fixed;
    "fixed": every part fixed (classical CMM kinematics), F_g only sets the volume J = det F_g;
    fields: {"<part>.<parameter>": value per Gauss point} instead of par (heterogeneous material);
    n_max > 0: FCMM mixture instead (cohort window n_max, homeostatic history at F = I; laws "wall", "removal", "fiber"
    without k_tau; mode unused)"""
    lib = fcmm if n_max else hcmm

    def make(Ge, Q, f, fl):
        e_r, e_t, e_z = Q[:, 0], Q[:, 1], Q[:, 2]
        cs = []
        for p in wall:
            q = lambda k: fl.get(f"{p.name}.{k}", par[f"{p.name}.{k}"])
            law = laws if p.laws == "turnover" else p.laws
            if law == "fiber":
                pr = fcmm.maes(q("T"), q("k_plus"), q("k_minus")) if n_max else hcmm.maes(q("T"), q("k_plus"),
                                                                                         q("k_minus"), q("k_tau"))
            elif law == "wall":
                k_turn = par.get(f"{p.name}.k_turn", 0.0)
                pr = (fcmm.MixtureProduction(q("T"), q("k_sigma"), k_turn) if n_max
                      else hcmm.MixtureProduction(q("T"), q("k_sigma"), q("k_tau"), k_turn)), lib.MixtureRemoval(q("T"), k_turn)
            elif law == "removal":
                pr = None, lib.MaesRemoval(q("T"), 0.0)
            else:
                pr = None, None
            rho = (f if p.lesion else 1 - f) * q("rho")
            fixed = "fixed" in (p.growth, growth)
            if p.material == "neo-hooke":
                inc = incompressible if p.incompressible is None else p.incompressible
                mat, G = materials.NeoHookeanInc(q("C10")) if inc else materials.NeoHookean(q("C10"), q("K")), Ge
            else:
                M = {"theta": e_t, "z": e_z}.get(p.direction)
                if M is None:
                    sign = 1.0 if p.direction == "+diag" else -1.0
                    M = jnp.sin(q("alpha")) * e_t + sign * jnp.cos(q("alpha")) * e_z
                if p.reorient:
                    pr = lib.Reorienting(pr[0], M), pr[1]
                mat, G = materials.Fung(q("k1"), q("k2"), M), G_fiber(q("g"), M)
            if n_max:
                cs.append(fcmm.constituent.allocate(fcmm.params(mat, G, *pr, grows=not fixed), rho,
                                                    mat.sigma_f(mat.sigma(G)), n_max, ds, filled=True))
            else:
                cs.append(hcmm.constituent(mat, rho, *pr, G=G, sigma_pre_mode=mode, growth=hcmm.Fixed() if fixed else None))
        g = lib.Isotropic() if growth in ("isotropic", "fixed") else lib.Anisotropic({"radial": e_r, "axial": e_z}[growth])
        return fcmm.mixture.allocate(cs, jnp.eye(3), ds, n_max, g, filled=True) if n_max else hcmm.mixture(cs, ds=ds, growth=g)

    f = jnp.zeros(G_e.shape[:2]) if lesion is None else lesion
    fl = {k: jnp.broadcast_to(v, G_e.shape[:2]).reshape(-1) for k, v in (fields or {}).items()}
    st = jax.vmap(jax.checkpoint(make))(G_e.reshape(-1, 3, 3), Qloc.reshape(-1, 3, 3), f.reshape(-1), fl)
    return jax.tree.map(lambda x: x.reshape(G_e.shape[:2] + x.shape[1:]), st)
