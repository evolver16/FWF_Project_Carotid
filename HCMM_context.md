# HCMM (Homogenized Constrained Mixture Model) — Implementation Context

Context handoff for continuing development of `HCMM.py` in JAX. Paste this into
`CLAUDE.md` or the start of a Claude Code session so the assistant doesn't have
to re-derive the architecture from scratch.

## Source material

- **Maes & Famaey (2023)**, *"How to implement constrained mixture growth and
  remodeling algorithms for soft biological tissues"*, J. Mech. Behav. Biomed.
  Mater. 140, 105733. The practical "how to implement" tutorial — this is what
  the code structure below follows most closely, including the accompanying
  Abaqus UMAT Fortran reference implementation (`UMAT_GR_HCM.for`,
  `UMAT_GR_HCM_NH.for`, `sigma_C_*.for`, `mnewt.for`/`fjacf.for` Newton solver).
- **Cyron, Aydin & Humphrey (2016)**, *"A homogenized constrained mixture (and
  mechanical analog) model for growth and remodeling of soft tissue"*, Biomech.
  Model. Mechanobiol. 15(6). The original continuous-time derivation of the
  homogenized theory (Eq. 1–25 in that paper). **Note: Cyron's paper contains
  no numerical/discretization scheme at all** — no Newton iteration, no
  time-stepping recipe. All of the numerical/implementation choices below come
  from Maes & Famaey's tutorial and its accompanying Fortran code, not from
  Cyron directly.

## Core design decision: `material` is stateless, `constituent` holds all state

- **`material`** (`Fung`, `NeoHookean`, JAX pytrees): pure functions of
  kinematics only. Exposes `sigma(F_e)`, `sigma_f(sigma)`, and a uniform
  `F_r(...)` method (tensor `F_r` in, tensor `F_r` out). Never stores any
  per-timestep state (no `lam_r`, no cached stress). This is required for
  `NeoHookean` (no scalar remodeling parametrization exists for a 3-D matrix)
  and is deliberately kept uniform for `Fung` too, so `constituent` doesn't
  need to special-case which material it holds.
- **`constituent`**: holds all persistent state — `rho`, `F_r`, `F_e`, `sigma`,
  `sigma_f`, `sigma_pre` (deposition prestress, set once at construction/
  homeostasis and never overwritten — only *rotated* per step, see below).

### Why `Fung` doesn't need to store `lam_r`

`Fung`'s remodeling ansatz (Eq. 38, Maes & Famaey) assumes `F_r` stays coaxial
with the fiber direction `M`: `F_r = λr·(M⊗M) + (1/√λr)·(I − M⊗M)`. This means
`λr` is trivially recoverable from `F_r` via `λr = M · (F_r @ M)` (since
`F_r @ M = λr · M` by construction) — a single dot product, negligible cost.
Storing `λr` separately from `F_r` would risk desync and leaks a
`Fung`-specific concept onto the generic `constituent`. **Always recover
`λr` from `F_r` inside `Fung.F_r`, never store it.**

### Why `NeoHookean` needs a genuine Newton solve

For an isotropic 3-D matrix (elastin), `F_r` has up to 6 independent unknown
components (symmetric tensor, compressible case — no `det(F_r)=1` constraint
since `NeoHookean` here is compressible, not incompressible). Eq. 20
(Cyron 2016) is then a coupled nonlinear system with no closed-form solution —
unlike the fiber case, there is no ansatz that collapses it to one scalar.
Maes & Famaey §2.4.2 ("three cases") is explicit that only the 1-D fiber case
has a closed form; 3-D compressible/incompressible always need Newton's
method. This is exactly why the reference Fortran carries `mnewt.for`/
`fjacf.for`/`ludcmp.for`/`lubksb.for` — but those are called *only* from
`UMAT_GR_HCM_NH.for` (the matrix material), never from the fiber UMAT.

## Governing equations (Maes & Famaey numbering unless noted)

- Eq. 15/17: `F_e^j(s) = F(s) @ inv(F_g(s)) @ inv(F_r^j(s))`
- Eq. 24 (density rate, forward-Euler'd via Eq. 37):
  `ρ̇+ = (ρ/T)·[1 + kσ+·(σf − σf,pre)/σf,pre]`,
  `ρ̇− = -(ρ/T)·[1 + kσ−·(σf − σf,pre)/σf,pre]`
- Eq. 17 (rotated deposition prestress): `σ_pre(s) = R(s) @ σ_pre(0) @ R(s).T`,
  where `R(s) = polar(F(s))` (rotational part only).
- Eq. 20 (Cyron 2016, tensor remodeling-rate equation):
  `(ρ̇+/ρ)·(σ − σ_pre) = (∂σ/∂F_e) : (F_e · L_r)`, with `L_r = Ḟr · Fr⁻¹`.
- Eq. 41 (closed-form fiber remodeling rate, projection of Eq. 20 onto `M`):
  `λ̇r = (ρ̇+/ρ)·(σf − σf,pre)·(Jφ)/(4ρλr) · [∂²Ψ/∂I4²·I4² + ∂Ψ/∂I4·I4]⁻¹`
- Eq. 43 (finite-difference approx of `Ḟr` for the 3-D Newton solve):
  `Ḟr ≈ (F_r(s+ds) − F_r(s)) / ds`

## Step ordering — the key architectural conclusion

**`F` is never lagged. `ρ` and `F_r` always are (by exactly one step).**

1. **Settle `F` for this step.**
   - Displacement-controlled (case U): `F` is prescribed directly, no solve.
   - Stress/force-controlled (cases S/F): outer Newton solve over `F`
     (3 unknowns for uniaxial single-element: `λx, λy, λz`), with residual
     built from `sigma(F_trial; F_r_old, rho_old)` — **using OLD `F_r`/`rho`,
     never re-solved per outer iterate**. This matches the reference UMAT:
     Abaqus's own equilibrium NR always reads `STATEV`'s committed (old)
     material state for stress/stiffness; growth/remodeling never feeds back
     into the equilibrium residual within the same increment (staggered
     scheme, not monolithic).
   - Residual for case S: `[sigma_xx − 0, sigma_zz − 0, sigma_yy − target]`
   - Residual for case F: same but 3rd component is nominal traction
     `P_yy·A0 = J·sigma_yy/λy·A0 − Force_target` (reference area effect).
2. **Compute `R = polar(F)`** — once per step, from the now-fixed `F`. Not
   inside any Newton loop (neither the outer `F` solve nor `F_r`'s inner
   solve) — nothing else in the step depends on `F_r` or perturbs `F`, so `R`
   is just a fixed input from here on.
3. **Rotate the deposition prestress**: `sigma_pre_s = R @ constituent.sigma_pre @ R.T`.
4. **Evaluate `sigma_now`, `sigma_f_now` at `(F, F_r_old)`** — i.e. build
   `F_e_old_basis = F @ inv(F_g) @ inv(constituent.F_r)` and evaluate
   `material.sigma(F_e_old_basis)`. **This uses the CURRENT step's `F`
   combined with the OLD `F_r`** — this is the one point that was repeatedly
   mis-stated during discussion and needs to be right: `sigma_f` is a function
   of current `F`, not a cached/lagged value from the previous step's `sigma`.
   (Only true equivalence when `F` is constant between steps, e.g. case U
   after displacement is fixed — NOT true in general for cases S/F.)
5. **Density update** (Eq. 24/37) using `sigma_f_now`.
6. **`F_r` update** — call `material.F_r(F_e_old_basis, sigma_now, sigma_pre_s,
   rho_dot_plus, rho_old, F_r_old, ds)`. Closed-form for `Fung`, Newton solve
   for `NeoHookean`. Single call, not nested inside step 1's outer loop.
7. **Recompute "settled" `sigma_new`, `sigma_f_new`** at
   `F_e_new = F @ inv(F_g) @ inv(F_r_new)` — this is what gets cached forward
   as `constituent.sigma` / `constituent.sigma_f` for next step's step-4 input.
8. Commit `rho_new, F_r_new, F_e_new, sigma_new, sigma_f_new` to a new
   `constituent` (functional update — pytrees are immutable, use a
   `_replace`-style helper).

## Staggered vs. monolithic scheme — deliberate choice, not forced

Recomputing `F_r_new(F_trial)` *inside* the outer `F`-Newton residual (fully
monolithic/coupled) would give a more mechanically self-consistent stress (no
"leftover" equilibrium error against the just-updated `F_r`), but:
- multiplies cost (inner Newton runs once per outer iterate instead of once
  total),
- requires differentiating through the inner solve (JAX can do this, but it's
  heavier — through-the-loop autodiff or implicit-function-theorem, vs. a
  simple `jax.jacfwd` on a fixed-`F_r` residual),
- no longer matches the reference Fortran's numbers,
- the error being removed (splitting error) is likely small anyway given
  `Δs ≪ T` in the paper's test cases (10 days vs. `T=101` days) — Fig. 3 of
  Maes & Famaey shows the homogenized theory already tolerates large step
  sizes well.

**Current design: staggered (current `F`, old `F_r`), matching the reference.**
Revisit only if empirical comparison against the paper's Fig. 2/3/4 results
shows meaningful disagreement attributable to splitting error.

## `Fung.F_r` (closed form)

```python
@jit
def F_r(self, F_e_old, sigma_f, sigma_pre_f, rho_dot_plus, rho, F_r_old, J, phi, ds):
    lam_r = jnp.dot(self.M, F_r_old @ self.M)          # inverse of Eq. 38
    I4 = self.I4(F_e_old)
    denom = self.d2Psi_dI4(I4) * I4**2 + self.dPsi_dI4(I4) * I4
    rate = (rho_dot_plus / rho) * (sigma_f - sigma_pre_f) * (J * phi) / (4 * rho * lam_r * denom)
    lam_r_new = lam_r + ds * rate
    return lam_r_new * self.P + (1.0 / jnp.sqrt(lam_r_new)) * (jnp.eye(3) - self.P)
```

## `NeoHookean.F_r` (Newton solve, RHS via `jax.jvp` — no explicit rank-4 stiffness)

```python
def sym_to_voigt(T):
    return jnp.array([T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2]])

def voigt_to_sym(v):
    return jnp.array([[v[0], v[3], v[4]],
                       [v[3], v[1], v[5]],
                       [v[4], v[5], v[2]]])

# inside NeoHookean:
@jit
def _residual(self, x, F_e_old, F_r_old, rhs, ds):
    F_r_new = voigt_to_sym(x)
    L_r = ((F_r_new - F_r_old) / ds) @ jnp.linalg.inv(F_r_old)   # Eq. 43, frozen at OLD Fr_inv
    _, dsigma = jax.jvp(self.sigma, (F_e_old,), (F_e_old @ L_r,))  # RHS of Eq. 20 == "Eq. 42 contracted with one direction"
    return sym_to_voigt(dsigma - rhs)

@jit
def F_r(self, F_e_old, sigma_old, sigma_pre, rho_dot_plus, rho, F_r_old, ds, n_iter=20):
    rhs = (rho_dot_plus / rho) * (sigma_old - sigma_pre)          # LHS of Eq. 20
    def newton_step(i, x):
        fvec = self._residual(x, F_e_old, F_r_old, rhs, ds)
        fjac = jax.jacfwd(self._residual)(x, F_e_old, F_r_old, rhs, ds)
        return x + jnp.linalg.solve(fjac, -fvec)
    x_final = jax.lax.fori_loop(0, n_iter, newton_step, sym_to_voigt(F_r_old))
    return voigt_to_sym(x_final)
```

Key point: `jax.jvp(self.sigma, (F_e_old,), (F_e_old @ L_r,))` computes
`(∂σ/∂Fe) : (Fe · L_r)` directly via forward-mode AD, without ever
materializing the full rank-4 stiffness tensor (Eq. 42 in the paper) — cheaper
than the Fortran reference's explicit Voigt-notation stiffness matrix
(`dsigdFe_e` in `sigma_C_iso_HCM.for`).

Deliberate deviation from reference: this uses a **full Newton** (Jacobian
re-evaluated every iterate via `jax.jacfwd`), whereas the Fortran `mnewt`/
`fjacf` freezes `Fe`, `DD` (stiffness) at the pre-remodeling values for the
whole inner loop (quasi-Newton). Full Newton is simpler/more robust to write
correctly in JAX; won't bit-for-bit match the reference on a single step but
should converge to the same fixed point.

Note: for `Psi(F) = C10·(I1bar−3) + K/2·(J−1)²` with `F, Fg = const`, `∂σ/∂F_e`
is exactly what Eq. 19/20's `|_{F,Fg=const}` subscript refers to — but since
`sigma(F_e)` (as written) is a pure function of `F_e` alone and never takes
`F`/`F_g` as separate arguments, the constraint is automatic: any autodiff
call on `sigma` is already implicitly "at constant `F`, `F_g`" because those
variables never entered the function to begin with.

## Open items / not yet resolved

- Should `Fung.F_r`'s signature be forced to match `NeoHookean.F_r`'s (full
  tensors `sigma`/`sigma_pre` in, `Fung` scalarizes internally via
  `self.sigma_f(...)`) so `constituent.step` never branches on material type?
  Currently `Fung.F_r` takes scalars (`sigma_f`, `sigma_pre_f`) directly.
- `constituent.step` needs a `_replace`-style functional-update helper (pytree
  nodes are immutable) — not yet written.
- Mixture-level orchestration (`F_g` calc, lagged per `J_g_calc`/`F_g_calc`
  already in `HCMM.py`, summing `sigma_tot = Σ φ_j·sigma_j`, Eq. 27) not yet
  wired to per-constituent `step`.
- Test cases U/S/F (§2.5 of Maes & Famaey, single-element, Fig. 1 BCs) not yet
  implemented as a driver loop.
- No verification yet against the paper's Fig. 2/3/4 numerical results.

## Files referenced during this discussion

- `HCMM.py` — current JAX implementation in progress (kinematics, `Fung`,
  `NeoHookean`, `constituent` skeleton)
- Fortran reference (Maes & Famaey's supplementary material): `UMAT_GR_HCM.for`
  (fiber), `UMAT_GR_HCM_NH.for` (elastin matrix, Newton solve), `mnewt.for`,
  `fjacf.for`, `ludcmp.for`, `lubksb.for` (Numerical Recipes Newton/LU solver),
  `sigma_C_HGO_C_fiber.for`, `sigma_C_iso_HCM.for`, `sigma_C_vol_HCM.for`,
  `sigma_C_HGO_C.for`, `sigma_C_iso.for`, `arguments_HCM_CM.for`
