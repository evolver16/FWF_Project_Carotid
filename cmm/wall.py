"""Wall composition: constituents as recipes (Part), assembled into HCMM states at every Gauss point."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

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
    k_minus, k_tau), "wall" (MixtureProduction: T, k_sigma, k_tau; constant removal), "removal" (constant removal, T);
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


@partial(jax.jit, static_argnames=("wall", "mode", "laws", "incompressible", "growth"))
def assemble(G_e, Qloc, par, wall, mode="deposition", ds=10.0, lesion=None, laws="fiber", incompressible=False,
             growth="radial"):
    """HCMM mixture per Gauss point from the parts of wall; Qloc = (e_r, e_theta, e_z) per Gauss point;
    laws, incompressible: defaults for the parts that do not set their own;
    growth: mixture F_g "radial" (hcmm.Anisotropic(e_r), thickening), "axial" (hcmm.Anisotropic(e_z), lengthening) or
    "isotropic" (hcmm.Isotropic); it also moves the stress-free state of every part whose growth is not fixed"""

    def make(Ge, Q, f):
        e_r, e_t, e_z = Q[:, 0], Q[:, 1], Q[:, 2]
        cs = []
        for p in wall:
            q = lambda k: par[f"{p.name}.{k}"]
            law = laws if p.laws == "turnover" else p.laws
            if law == "fiber":
                pr = hcmm.maes(q("T"), q("k_plus"), q("k_minus"), q("k_tau"))
            elif law == "wall":
                pr = hcmm.MixtureProduction(q("T"), q("k_sigma"), q("k_tau")), hcmm.MaesRemoval(q("T"), 0.0)
            elif law == "removal":
                pr = None, hcmm.MaesRemoval(q("T"), 0.0)
            else:
                pr = None, None
            rho = (f if p.lesion else 1 - f) * q("rho")
            own = hcmm.Fixed() if p.growth == "fixed" else None
            if p.material == "neo-hooke":
                inc = incompressible if p.incompressible is None else p.incompressible
                mat = materials.NeoHookeanInc(q("C10")) if inc else materials.NeoHookean(q("C10"), q("K"))
                cs.append(hcmm.constituent(mat, rho, *pr, G=Ge, sigma_pre_mode=mode, growth=own))
            else:
                M = {"theta": e_t, "z": e_z}.get(p.direction)
                if M is None:
                    sign = 1.0 if p.direction == "+diag" else -1.0
                    M = jnp.sin(q("alpha")) * e_t + sign * jnp.cos(q("alpha")) * e_z
                if p.reorient:
                    pr = hcmm.Reorienting(pr[0], M), pr[1]
                cs.append(hcmm.constituent(materials.Fung(q("k1"), q("k2"), M), rho, *pr, G=G_fiber(q("g"), M),
                                           sigma_pre_mode=mode, growth=own))
        g = hcmm.Isotropic() if growth == "isotropic" else hcmm.Anisotropic({"radial": e_r, "axial": e_z}[growth])
        return hcmm.mixture(cs, ds=ds, growth=g)

    f = jnp.zeros(G_e.shape[:2]) if lesion is None else lesion
    st = jax.vmap(jax.checkpoint(make))(G_e.reshape(-1, 3, 3), Qloc.reshape(-1, 3, 3), f.reshape(-1))
    return jax.tree.map(lambda x: x.reshape(G_e.shape[:2] + x.shape[1:]), st)
