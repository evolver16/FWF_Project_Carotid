"""G&R studies, run from cmm/:  python -m studies <study> [positional ...] [key=value ...]
Outputs in results/<group>/<study>/.

material  exact_solution      HCMM / FCMM vs the exact constrained mixture solution at constant F
          maes_benchmark      Maes & Famaey (2023) models A-E, cases U/S/F, Fig. 2
          parameter_sweep     HCMM-FCMM gap over prestretch, gain and step size
          case_comparison     one model / case, HCMM vs FCMM over time (plot)
          fe_readiness        batching, objectivity, tangent and gradients of the models for the FE code
fe_gr     fe_vs_material_point  HCMM patch test, single (hybrid) elements vs the material point, hybrid adjoint
          element_formulations  artery G&R with standard / F-bar / hybrid / tet10, tet10 adjoint, inclined supports
artery    artery_gr           G&R of a quarter cylinder (Maes & Famaey 2023 Fig. 4)
          artery_gradients    adjoint gradients vs finite differences, parameter identification, cost
          pressure_step       pressure step held constant: elastic vs turnover widening
          pressure_buckling   pressure buckling of a G&R state (G&R frozen): critical pressure, post-buckling path
vessel    stenotic_vessel     bent stenotic vessel from an .inp mesh, Laplace wall basis, adjoint
          bending_redistribution  mass redistribution in a bent tube (end rotation)
          setpoint_patch      lowered collagen set point in a patch: growth-only displacement
"""

from studies import artery, fe_gr, material, vessel

STUDIES = {f.__name__: f for f in (
    material.exact_solution, material.maes_benchmark, material.parameter_sweep, material.case_comparison,
    material.fe_readiness,
    fe_gr.fe_vs_material_point, fe_gr.element_formulations,
    artery.artery_gr, artery.artery_gradients, artery.pressure_step, artery.pressure_buckling,
    vessel.stenotic_vessel, vessel.bending_redistribution, vessel.setpoint_patch,
)}
