"""G&R studies, run from cmm/:  python -m studies <study> [positional ...] [key=value ...]
Outputs in results/<group>/<study>/.

material  exact_solution      HCMM / FCMM vs the exact constrained mixture solution at constant F
          maes_benchmark      Maes & Famaey (2023) models A-E, cases U/S/F, Fig. 2
          parameter_sweep     HCMM-FCMM gap over prestretch, gain and step size
          case_comparison     one model / case, HCMM vs FCMM over time (plot)
          fe_readiness        batching, objectivity, tangent and gradients of the models for the FE code
          maes_abaqus         Fig. 2 from Maes' Abaqus UMATs, the same UMATs in fem.System and our HCMM / FCMM
          maes_timing         run time per Maes case: Abaqus, UMAT in fem.System, our models in fem.System / point
fe_gr     fe_vs_material_point  HCMM patch test, single (hybrid) elements vs the material point, hybrid adjoint
          element_formulations  artery G&R with standard / F-bar / hybrid / tet10, tet10 adjoint, inclined supports
          solution_verification  mesh and time step convergence of the artery G&R (GCI)
artery    artery_gr           G&R of a quarter cylinder (Maes & Famaey 2023 Fig. 4)
          artery_gradients    adjoint gradients vs finite differences, parameter identification, cost
          pressure_step       pressure step held constant: elastic vs turnover widening
          flow_step           flow step, WSS stimulus from the deformed lumen (Poiseuille stand-in for a fluid solver)
          pressure_buckling   pressure buckling of a G&R state (G&R frozen): critical pressure, post-buckling path
vessel    stenotic_vessel     bent stenotic vessel from an .inp mesh, Laplace wall basis, adjoint
          bending_redistribution  mass redistribution in a bent tube (end rotation)
          stenosis_wss        WSS-driven remodeling of a bent stenosis (flow as BC.wss, ParaView series)
          setpoint_patch      lowered collagen set point in a patch: growth-only displacement
weiss     weiss_tortuosity    Weiss et al. 2020: tortuosity from an Eq. 8 insult (c_e, alpha_0), isometric vs isotonic ends
          free_angle          Hariton 2007 collagen angle from the stresses on the straight tube: homeostasis, elastin loss
          weiss_rod           weiss_tortuosity as a 3D rod with HCMM ring sections (reduced model)
carotid   bifurcation         idealized carotid bifurcation: mesh, Laplace wall basis, homeostatic prestress
          junction_surface    bifurcation as in CAD: CCA + half-ball, ICA / ECA from the rim, rolling-ball rounds
          bifurcation_tortuosity  Weiss wall on the bifurcation: elastin loss, emergent collagen angle, ICA tortuosity
          bifurcation_rod     bifurcation_tortuosity as CCA / ICA / ECA rods with HCMM ring sections (reduced model)

Shared by several studies: common (logging, outputs), model_e (model E artery: artery_par, WALL_E, artery_simulate).
Study-specific walls and parameters live in the study; a study with several files gets its own subfolder.
"""

from studies import artery, carotid, fe_gr, material, vessel
from studies.weiss import free_angle, rod
from studies.weiss import tortuosity as weiss

STUDIES = {f.__name__: f for f in (
    material.exact_solution, material.maes_benchmark, material.parameter_sweep, material.case_comparison,
    material.fe_readiness, material.maes_abaqus, material.maes_timing,
    fe_gr.fe_vs_material_point, fe_gr.element_formulations, fe_gr.solution_verification,
    artery.artery_gr, artery.artery_gradients, artery.pressure_step, artery.flow_step, artery.pressure_buckling,
    vessel.stenotic_vessel, vessel.stenosis_wss, vessel.bending_redistribution, vessel.setpoint_patch,
    weiss.weiss_tortuosity, free_angle.free_angle, rod.weiss_rod,
    carotid.bifurcation, carotid.junction_surface, carotid.bifurcation_tortuosity, carotid.bifurcation_rod,
)}
