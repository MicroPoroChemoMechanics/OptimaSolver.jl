# Changelog

## v0.3.0 — a primal-dual interior-point method

### Breaking changes

No name was removed or renamed, and every public signature is unchanged. The
bump is a minor one because **the numbers move**: this replaces the primal
barrier iteration with a primal-dual one, so a solve returns a different — and
much better — iterate. Below 1.0 the resolver treats a minor bump as breaking
regardless, so a downstream `[compat] OptimaSolver = "0.2"` must be widened.

### The bound multipliers are now independent variables

The method substituted `z = μ/s` for the bound multipliers. That is the textbook
primal barrier method, and it has a textbook failure mode: as a variable
approaches its bound, `μ/s` diverges. Measured on a cement equilibrium, the
optimality residual started at 4.5e11 and never fell below 2e9, so
`is_converged` could not fire at ANY tolerance — `tol = 1e-4` failed exactly as
`tol = 1e-10` did. The barrier therefore never fell from its initial 1e-4, the
step was capped by the fraction-to-boundary rule at every single iteration, and
the solver ran to `max_iter` and stopped wherever it happened to be.

`z` is now carried explicitly, with its own Newton step
`dzᵢ = (μ − sᵢzᵢ − zᵢdnᵢ)/sᵢ`, its own fraction-to-boundary limit, and a
central-path box that keeps it from drifting away from `μ/s` and poisoning the
curvature `Σ = Z S⁻¹`. The Hessian seen by the step is `hf + z/s` rather than
`hf + μ/s²`, so it tracks the actual multiplier instead of the barrier
parameter.

Convergence is measured on the two conditions that must both vanish — dual
feasibility scaled by the slack, and complementarity `sᵢzᵢ − μ` — rather than on
the raw Newton residual.

### What it buys

On the equilibrium partition of a full ordinary Portland cement, replayed at
seven instants over 28 days, the element balance goes from

| instant | before | after |
|:--|--:|--:|
| 0.05 d | 5.3e-3 | 3.3e-9 |
| 0.25 d | 3.3e-4 | 7.1e-15 |
| 1 d | 1.4e-7 | 1.8e-15 |
| 28 d | 5.0e-9 | 2.8e-11 |

with the worst over the 201 accepted steps of the coupled run falling from
4.3e-4 mol to 6.0e-6, and the median to 2.7e-9. The six-hour instant — the AFt
peak, where the assemblage switches and which had resisted every other remedy —
is now solved to machine precision.

### Also fixed

- A dead test dependency on ChemistryLab, pinned at `"0.2, 0.3"` while no test
  referenced it. It closed a dependency loop, ChemistryLab depending on this
  package.

### Still open

The solver does not report `converged` on a cement equilibrium even now: the
iterate is right and the balance is at machine precision, but the KKT error does
not cross `tol`. Judge these solves on the element balance, not on the return
code.

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
