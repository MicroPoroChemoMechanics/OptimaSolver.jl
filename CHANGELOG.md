# Changelog

## v0.3.0 — a KKT solver that proves its answer

### Breaking changes

Five new exported names — `DualNewtonProblem`, `DualNewtonOptions`,
`dual_newton_solve`, `kkt_certificate`, `degenerate_components`. Nothing was
removed or renamed and no existing signature changed, but below 1.0 the resolver
treats a minor bump as breaking regardless, so a downstream
`[compat] OptimaSolver = "0.2"` must be widened.

### `dual_newton_solve`: Newton on the KKT system in multiplier space

The interior-point method of `solve!` minimises `f` by walking the interior, and
on a problem whose bounded variables have zero curvature it does not reach its
tolerance: the fraction-to-boundary rule caps the step at every iteration. That
is intrinsic to a primal barrier method.

The alternative solves the KKT conditions directly. With `u = −Aᵀy`, an interior
variable obeys `hᵢ(x) = uᵢ − gᵢ` — invertible, a mass-action law in a chemical
system — and a bounded variable is positive exactly when `gᵢ = uᵢ`, zero when
`gᵢ ≥ uᵢ`. That second line is a stability criterion, and the active set on it is
finite for a convex problem.

Parameterising the interior variables by `ln x` makes their positivity automatic,
so the fraction-to-boundary rule has nothing left to act on. The outer system is
`1 + m + |P|` unknowns, some fourteen for a cement partition against forty-seven
variables in the interior-point route.

This is **not** the log reparameterisation of `variable_space = Val(:log)`:
`f ∘ exp` has second derivative `xᵢ(∇fᵢ + 1)`, negative wherever `∇fᵢ < −1`,
hence not convex. The logarithm is applied to the KKT *equations*, and convexity
of the original problem is what makes their solution unique.

Three points had to be right:

  - **a variable whose `hᵢ` is bounded above cannot be inverted.** In a chemical
    system that is the solvent, whose activity is a mole fraction, so `ln a ≤ 0`
    always and an arbitrary `y` may demand more, for which no finite `x` exists.
    It is declared through `j_ref` and carried by the outer system. An inner loop
    that included it could never report convergence, which then invalidated the
    outer Jacobian, that Jacobian being derived on the assumption the inner
    conditions hold exactly.
  - **the active set must change during the Newton, one variable at a time.** Two
    variables both declared stationary over-determine `y` and their rows are
    jointly infeasible; admitting a batch feeds a cycle. Visited sets are
    recorded, which bounds the loop by the number of subsets and therefore
    terminates. Observed on a cement without limestone as a solve converged to
    `2e-12` — of the wrong subproblem, an excluded variable violated by 10.9.
  - **`bₖ = 0` does not mean degenerate.** With `x ≥ 0` the row forces its
    variables to vanish only when its non-zero entries share a sign;
    `degenerate_components` implements that test. The `H+` row of a chemical
    system carries `+1` for `H+` and `−1` for `OH-`, so its zero total is the
    ordinary state of pure water — treating it as degenerate removes the entire
    acid–base system and returns pH 7.000 with the solid undissolved.

### `kkt_certificate`: a proof, not a plausibility argument

For a convex program the KKT conditions are sufficient, so checking them settles
optimality. The check reports the stationarity of the interior variables, the
feasibility of the equalities, and the worst violation among variables at their
bound.

Two splits decide whether it means anything: a variable at its bound obeys the
inequality, not the equality — imposing the equality on an amount held at `1e-16`
whose stationarity value is `e⁻³⁰⁰` misstates `hᵢ` by 263 units, and the check
then reports 74 for a point solved to `5e-12` — and a variable carrying a
degenerate component is excluded from both tests.

Used through ChemistryLab on its Reaktoro reference, the certified answer matches
**every** species to 1 %, including one the reference test records as
`@test_broken` because the interior-point answer is 147 % high. On calcite in
pure water the certified pH is 9.90 against an interior-point 6.96.

### Fixed

- **`is_converged` could never fire on a chemical equilibrium.** The optimality
  error was the Newton residual `g_L − μ/s`, which diverges at the bounds:
  writing `s = s*(1+η)`, it equals `(μ/s*)·η/(1+η)`, unbounded as `s* → 0` at
  fixed *relative* error. No tolerance was attainable — `tol = 1e-4` failed
  exactly as `tol = 1e-10` did. The error began at `4.5e11` and never fell below
  `2e9`, the barrier therefore never fell from its initial `1e-4`, and every
  solve ran to `max_iter` while the feasibility error was reaching `1e-14`. The
  guard meant to prevent this, excluding variables within `1e-6 × lb` of their
  bound, could not fire either: with `lb = 1e-16` that threshold is `1e-22`.

  Stationarity is now measured in complementarity form, `Eᵢ = sᵢ·g_{L,i} − μ`,
  obtained by eliminating the bound multiplier through dual feasibility. Same
  zeros, bounded where the residual form is not; on the same iterate it reads
  `5.8e-3` instead of `4.5e11`.

- **The warm-start cache overrode the caller's initial point.** It is now
  consulted only when the problem has the same size *and* `u0` carries no
  interior information. A kinetics run leaves its final composition in the cache,
  so replaying the same trajectory through the same algorithm object started
  every solve from the end state, returning pH 14.2 where honouring the guess
  gives 12.58.

- **A dead test dependency on ChemistryLab**, pinned at `"0.2, 0.3"` while no
  test referenced it; it closed a dependency loop.

### Documentation

The theory page now proves the convexity of the objective — the ideal mixing
Hessian is `diag(1/x) − 11ᵀ/N`, positive semidefinite by Cauchy–Schwarz, and the
bounded variables enter linearly — derives the failure of the residual form and
the equivalence and boundedness of the complementarity form, and states the
KKT-space formulation with its three subtleties and its termination argument.

## v0.2.8 — the optimality error was unattainable by construction

### Fixed

- **`is_converged` could never fire on a chemical equilibrium.** The optimality
  error was `‖∇f + Aᵀy − μ/s‖∞`, the Newton residual itself. As a variable
  approaches its bound that quantity diverges: with `μ = 1e-4` and `s = 1e-15`,
  `μ/s = 1e11`. On a cement equilibrium `err_opt` began at **4.5e11** and never
  fell below 2e9, so no tolerance could be met — `tol = 1e-4` failed exactly as
  `tol = 1e-10` did. Two consequences followed silently: `should_reduce_barrier`
  never let `μ` fall from its initial 1e-4, and every solve ran to `max_iter` and
  stopped wherever it happened to be, while the *feasibility* error was
  meanwhile reaching 1e-14.

  The guard meant to prevent this — excluding variables within `1e-6 × lb` of
  their bound — could not fire either: with `lb = 1e-16` that threshold is
  `1e-22`, so nothing was ever excluded. It was the exact opposite of the defect
  its own comment describes fixing.

  Stationarity is now measured in complementarity form, `sᵢ(∇f + Aᵀy)ᵢ − μ`,
  which is the equivalent condition multiplied through by `sᵢ`: bounded, zero at
  the optimum, and the measure Ipopt reports (Wächter & Biegler 2006, §3.5). On
  the same cement equilibrium `err_opt` is **5.8e-3** instead of 4.5e11, and the
  feasibility error reaches 1.4e-14. The Newton step is unchanged — only the
  error norm is.

### What this does not fix

The solver still does not reach its tolerance on a cement equilibrium. With the
error now meaningful, the trace shows why: the iteration is **non-monotone** and
the line search collapses, `α` falling to 2e-4 while `err_opt` stalls near
5.8e-3 and at times increases. Running longer can make the answer worse before
it makes it better. A full ordinary Portland cement coupling still shows element
imbalances up to 1.1 mol at its worst early steps, though it closes to 1e-10 mol
from three days on. That is a line-search and barrier-update problem, and it is
not addressed here.

## v0.2.7 — the warm-start cache no longer overrides the caller's guess

### Fixed

- **`OptimaOptimizer` silently discarded an explicit `u0`.** The algorithm object
  carries `_cache`, the previous solution, and with `warm_start = true` (the
  default) it started every solve from that cache — even when the caller had
  supplied a deliberate starting point, and even when the cached solution came
  from a *different* problem that the same algorithm object happened to solve
  earlier.

  The consequence is not academic. A chemical-kinetics run re-speciating at
  every accepted step leaves the final composition in the cache; replaying the
  same trajectory through the same algorithm object then starts every solve from
  the end state. On an ordinary Portland cement that returned a pore solution at
  pH 14.2 with 0.31 mol of ettringite and no monosulphate, where honoring the
  caller's guess gives pH 12.58 with the sulfate entirely in monosulphate — same
  trajectory, same constraints, same guess. It also quietly defeated the caller's
  own warm-start logic during the run itself.

  The cache is now what it was meant to be: a convenience for repeated solves
  where the caller has nothing better to offer. It is consulted only when the
  problem has the same size **and** `u0` carries no interior information, i.e.
  every variable still sits at its lower bound. A caller who supplies a real
  starting point now gets it.

  No API changed and nothing was removed. `reset_cache!` keeps its meaning.

- **A dead test dependency on ChemistryLab is removed.** It sat in `[extras]`,
  `[targets]` and `[compat]` pinned at `"0.2, 0.3"` — six minor versions behind —
  while no test file referenced it. It also closed a dependency loop, since
  ChemistryLab depends on this package: resolving the test environment demanded
  an ancient ChemistryLab that cannot coexist with the current one.

## v0.2.6 — The nullspace step no longer throws

### Fixed

- **`PosDefException` from the default code path.** The nullspace step factorizes
  the reduced Hessian `Zᵀ H Z` by Cholesky. That matrix is positive definite at
  any interior point in exact arithmetic, since `h > 0` there whatever the
  curvature — but it can lose definiteness numerically when the amounts span ten
  orders of magnitude. Hit on a calcite-dissolution trajectory where a species
  sits at `1e-16`.

  The factorization is now attempted (`check = false`) and falls back to a
  Bunch–Kaufman factorization, which handles the indefinite case. A default path
  must not throw on a well-posed problem.

## v0.2.5 — Nullspace Newton step, and the water autoprotolysis

### Changed (default behavior)

- **The Newton step is computed by the nullspace method by default**
  (`nullspace_step = true`). The previous route formed the Schur complement
  `S = A H⁻¹ Aᵀ`, which needs `H` invertible — and in a chemical equilibrium a
  **pure phase** has unit activity, hence `∂²G/∂nᵢ² = 0` exactly. The step then
  degenerates and the solve settles on a point that is not the minimum.

  The nullspace method writes `dn = dnₚ + Z dz` with `Z` a basis of `null(A)`,
  and since `Zᵀ Aᵀ = 0` the dual drops out of the projected stationarity:

      (Zᵀ H Z) dz = −Zᵀ (ex + H dnₚ)

  where `H` appears only as a *product*. The canonicalizer already supplies the
  basis: with `R = B⁻¹N`, `Z[jb, :] = −R` and `Z[jn, :] = I`, so
  `Zᵀ H Z = Rᵀ diag(h[jb]) R + diag(h[jn])`, symmetric positive definite at any
  interior point. This is the route the C++ Optima this package is ported from
  takes by default; its `Rangespace` counterpart — the Schur complement — is
  documented there as suitable for invertible diagonal Hessians only.

  Measured: pure water comes out at `[H⁺]/[OH⁻] = 1.0` and `pKw = 13.9994`,
  against 3.78 and 13.9897 through the Schur complement. Mixed solid/aqueous
  systems are unchanged. Set `nullspace_step = false` to restore the old step.

- **The convergence test excluded trace species from the stationarity check.**
  A variable was judged "at its bound" when its slack fell below
  `1e-8 × max_slack`, a threshold scaled by the *largest* variable in the
  problem. In an aqueous system the solvent sits at 55 mol, so the threshold
  became `5.5e-7` and every trace ion below it was declared to sit on a bound of
  `1e-16` — nine orders of magnitude away — and its optimality residual was
  never enforced. The criterion is now relative to the variable's own bound,
  which is what "sitting on it" means.

  Measured on calcite + CO₂ against Reaktoro: `CaOH⁺` ×20.8 → ×2.47, `OH⁻`
  ×3.19 → ×1.045, `H⁺` ×1.24 → ×1.006, and `pKw` 13.40 → 13.979. Every species
  except `CaOH⁺` now agrees to 5 % or better, and Ipopt lands on the same
  `CaOH⁺` value (×2.46), so that residual belongs to neither back-end.

### Added

- **The exact Hessian diagonal may be handed over through the problem
  parameters**, alongside `A` and `b`: a `hdiag` entry in the parameter
  `NamedTuple`, called as `hdiag(hf, n)`. Useful when the caller can compute
  `∂²f/∂nᵢ²` analytically instead of leaving the back-end to approximate it.

### Changed

- **`use_fd_hessian` has opposite defaults on the two ways of building an
  optimizer** — `false` on `OptimaOptions`, `true` on the `OptimaOptimizer(; …)`
  keyword constructor — so the same optimizer built two ways selects different
  regimes. The behavior is unchanged; both are now documented.

## v0.2.4 — Dual numbers cross the solve

### Fixed

- **`line_search` rejected dual numbers.** Its `α_max` keyword was annotated
  `::Float64`, while the value handed to it comes from `clamp_step`, which
  carries the type of the iterates. Differentiating through a solve therefore
  stopped with `TypeError: in keyword argument α_max, expected Float64`. The
  annotation is dropped and `α_max` now enters the `Tv` promotion alongside the
  iterates and the barrier parameter.
- **`clamp_step` carried the same annotation on `τ`**, and left it out of its
  own promotion. No internal caller passes a dual `τ`, so this one never failed
  in practice; it is fixed alongside `α_max` because one feeds the other and the
  asymmetry would invite the bug back.

Nothing else changes: `α_max` and `τ` were already converted through `Tv(...)`
and `T`, so the generic path was written and merely blocked at the door.

### Added

- Regression test covering a full `solve` driven by a dual-valued `b`, checked
  against the analytic `sensitivity`. The existing AD tests differentiated the
  building blocks (`kkt_residual`, `hessian_diagonal`, `gibbs_hessian_diag`)
  but never the Newton loop, which is why the above went unnoticed.

## v0.2.3 — Maintenance

- GitHub is now the sole home: the Codeberg return path (`.forgejo/` workflows
  and `docs/deploy_docs.jl`) is removed.
- US/UK spelling consistency check added.
- CI installs the General registry explicitly, working around Pkg server issues.
- No API changes, no functional changes.

## v0.2.2 — Maintenance

- Maintenance release: no API changes, no functional changes.
- CI badge restored; Runic badge.
- Installation instructions updated for registration in Julia's General
  registry (no registry to add beforehand).

## v0.2.1 — Maintenance

- Maintenance release: no API changes, no functional changes

## v0.2.0 — Packaging & relicensing

- Relicensed to LGPL-2.1-or-later
- Registered in MPCM-Registry
- GitHub Actions workflows: CI, Documentation, Register, CompatHelper, TagBot
- Multi-version documentation deployment (`docs/deploy_docs.jl`)

## v0.1.0 — Initial release

Primal-dual interior-point solver for Gibbs-energy minimization.

- `OptimaProblem` / `OptimaOptions` / `OptimaResult` API
- Schur-complement Newton step exploiting diagonal Hessian structure
- Filter line search (Wächter & Biegler 2006)
- Implicit-differentiation sensitivity matrices (`∂n*/∂b`, `∂n*/∂μ⁰`)
- SciML drop-in via `OptimaOptimizer` / `SciMLBase.solve`
- ForwardDiff-compatible throughout
