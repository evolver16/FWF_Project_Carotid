"""HCMM constituent materials, one per file. Interface: sigma(F_e), sigma_f(sigma) and
F_r(F_e, J, c, sigma_rate_euler) -> F_r(s+ds) for constituent c (Eq. 20)"""

from hcmm_materials.fung import Fung
from hcmm_materials.neo_hookean import NeoHookean
from hcmm_materials.neo_hookean_inc import NeoHookeanInc
from hcmm_materials.voigt import sym_to_voigt, voigt_to_sym
