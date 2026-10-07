"""Constituent materials shared by HCMM and FCMM, one per file. Interface: sigma(F_e), sigma_f(sigma);
HCMM also: F_r(F_e, J, c, sigma_rate_euler) -> F_r(s+ds) for constituent c (Eq. 20);
hcmm.core_implicit optional: flow_basis (remodeling directions), flow_exp(z) = exp(sum_i z_i B_i) in closed form"""

from materials.fung import Fung
from materials.neo_hookean import NeoHookean
from materials.neo_hookean_inc import NeoHookeanInc
from materials.voigt import sym_to_voigt, voigt_to_sym
