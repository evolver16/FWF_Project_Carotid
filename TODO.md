# TODO

Open in VS Code with `Ctrl+Shift+V` for the rendered view. Tick with `[x]`, add items with `- [ ]`.
Links are relative to this file and clickable in the preview.

## 1. Mechanobiology: constituents and laws

- [ ] Identify the mechanobiologically relevant constituents of the carotid wall (elastin, collagen families, smooth muscle, ground matrix, ...)
- [ ] Decide which laws we want per constituent:
  - growth (F_g: isotropic / along e_r)
  - remodeling (turnover-driven F_r)
  - production (stimulus: intramural stress, wall shear stress, ...)
  - removal (constant half-life, stress- or stretch-dependent, elastin degradation)
- current laws: [hcmm/production.py](cmm/hcmm/production.py), [hcmm/removal.py](cmm/hcmm/removal.py), [hcmm/growth.py](cmm/hcmm/growth.py); literature overview: [docs/gr_literature.html](docs/gr_literature.html)

## 2. Vessel geometry

- [ ] Carotid geometry (segmentation, bifurcation, wall thickness, layers), mesh import exists: [fem/mesh.py](fem/mesh.py)

## 3. Prestretch / in-vivo axial stretch

- [ ] Decide how the in-vivo stretch (carotid ~1.6 axially) enters: elastin deposition stretch, end BC (force or displacement), or both

**Our solution** ([artery_simulate](cmm/studies/artery.py)):
- reference configuration = imaged in-vivo geometry; collagen and SMC keep their deposition stretch (turnover maintains it)
- only the elastin prestretch is unknown: fixed point G_e <- F G_e (Anderson-accelerated) at the in-vivo pressure until max|u| < 1e-6 mm
- pointwise full tensor at every Gauss point, exact equilibrium, any geometry, no artificial rollers, differentiable
- non-unique (6 unknowns, 3 equilibrium equations per point): picks the solution closest to the start value (`g_ax`, `G_θ`);
  in a straight tube with axially fixed ends the axial component stays at `g_ax`, in curved geometry it can change
- idea: virtual excision (remove pressure and axial constraint), fit `g_ax` so the retraction matches the measured 1/λ_iv

**Laubrie et al. 2022** ([.context/Papers](.context/Papers)):
- also only the elastin prestretch is unknown; collagen and SMC at fixed deposition stretch (1.1)
- elastin prestretch diagonal in the local frame with a linear gradient from inner to outer curvature (torus approximation)
- axial value at the outer curvature fixed to the in-vivo axial stretch; inner axial and circumferential values adjusted
- 3 scalars iterated with forward FE runs until distortion < 3 % (thickness) and < 6 % (diameter): equilibrium only approximate,
  springs / radial rollers at the ends
- conclusion: the prestretch distribution changes the G&R response, it should be identified regionally from data

## 4. Verification

**Done so far** (all set up and run by Claude, against analytical / manufactured solutions and internal consistency; logs in
[verification/results/](verification/results/), overview in [docs/fem.html](docs/fem.html) section 16):
- patch tests, homogeneous deformations, Lamé tube, finite-strain incompressible tube, Euler column, arch snap-through
- manufactured solutions (convergence orders of all element formulations, incl. hybrid P2/P1 and J_target)
- consistency: tangent = AD of the residual, Newton order, follower pressure, invariances, adjoint vs finite differences
- Cook's membrane (locking), G&R mesh / time-step convergence (GCI), FE vs material point

**To do:**
- [ ] Review the verification scripts and results independently ([verification/](verification/))
- [ ] Compare with Abaqus (same mesh, BCs, material; `.inp` export exists: `mesh.write_inp`)
  - element mapping: tet10 standard = C3D10, hybrid_p1 ≈ C3D10H (linear pressure), hybrid hex8 ≈ C3D8H,
    Abaqus C3D8 uses selectively reduced integration (closer to our fbar than to standard)
  - cases: Lamé / finite-strain tube, patch tests, Cook's membrane, arch
- [ ] G&R against the Maes UMATs ([.context/FortranFiles](.context/FortranFiles); needs Abaqus + Fortran compiler)

## 5. End point: modeling workflow

- [ ] Explicit HCMM ([hcmm/core.py](cmm/hcmm/core.py)): main model, fast
- [ ] Parameter identification on the explicit HCMM (adjoint gradients, [artery_gradients](cmm/studies/artery.py))
- [ ] Implicit HCMM ([hcmm/core_implicit.py](cmm/hcmm/core_implicit.py)) with the identified parameters: more accurate, larger time steps
- [ ] FCMM ([fcmm/](cmm/fcmm/)) for verification of the homogenized results
