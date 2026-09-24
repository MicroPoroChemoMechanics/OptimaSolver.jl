# Changelog

## v0.6.1 — an inner iteration that has stopped converging no longer runs to its cap

The inner fixed point of the dual Newton, `_invert_phases!`, which recovers the
phase compositions from the multipliers, stopped only when its largest step fell
below `1e-14` or after 200 sweeps. On a 107-species cement solved cold from its
clinker composition, 89 % of the 69 463 calls ran to that cap, and they were not
approaching the threshold: their final step had a median of 30, which is the
bound on a single update, and the step at sweep 200 was the step at sweep 100.
These calls come from trial points of the line search and from the early
iterates of a cold start, where the multipliers ask for compositions that no
state within the bounds satisfies. The update then oscillates against its
clamp, and every such call paid 200 sweeps to return a state that the line
search went on to refuse.

The iteration now also stops once its step has gone `INNER_STALL_SWEEPS = 20`
sweeps without reaching a new minimum. A converging iteration reaches a new
minimum at every sweep, however many sweeps it needs, and is left untouched;
only an iteration that has ceased to contract is interrupted, and since such an
iteration ends above `inner_tol` in either case, no accepted step depended on
the sweeps that are removed. A test pins the three regimes on a real map
`w ← (1 − β) w`: a cycle stops after 21 sweeps, a contraction needing 49 sweeps
is not cut, and a monotone convergence too slow for 200 sweeps still runs to the
cap.

Measured with the new `benchmark/run.jl`, on one machine:

| solve | 0.6.0 | 0.6.1 |
|:--|--:|--:|
| cement, cold | 142.6 s | 16.8 s |
| cement, cold, first call | 165.0 s | 37.6 s |
| cement, warm from its answer | 1.00 s | 0.17 s |
| cement, neighboring budget, warm | 1.01 s | 0.17 s |

Every answer remains certified, and the largest relative change of an amount
above `1e-10` mol is `4.1e-13`. The sweeps fall from 12.8 million to 1.40
million; the calls barely change, from 69 463 to 63 661, because the saving is
in the cost of each non-converging call and not in their number.

The number of calls is where the choice of window shows, and it should not be
read as a property of the rule. The state returned by a stalled call is not the
one the two-hundredth sweep would have produced, and the outer Newton follows
it: with a window of 10 the same solve needed 492 calls and certified on its
first route in 0.21 s, with 25 it needed 28 150 and took 20.5 s. The
composition agreed to `3e-10` in all three cases. The value 20 rests on the
argument above and was not selected as the fastest of these runs.

### Added — a benchmark that judges time and answer together

`benchmark/` times the certified route on a frozen cement and compares each
answer with a recorded baseline, so that a faster solver returning a different
composition shows up as the regression it is. `benchmark/sweeps.jl` counts the
sweeps of the inner iteration without instrumenting the library: it reads the
method from the source, grafts a counter onto its loop and replaces it for that
session only.

### Downstream — ChemistryLab

One ChemistryLab test, in `test/diffuse_layer.jl`, pins the size of a failure
that this release reduces. Above `ELECTROSTATIC_STIFFNESS_LIMIT`, the route that
eliminates the surface potential used to diverge with a stationarity above
`1e-3`; with this release it stops between `1e-8` and `3e-7` on the two affected
points, still refused by the certificate, and the stiffness bracket that the
test exists to establish still holds. The test has to classify its points by
the certificate before ChemistryLab resolves this version, which its compat
bound `"0.5.5, 0.6"` will do automatically.

## v0.6.0 — the stability test now says WHERE, and a rigorous answer to "is this budget possible at all?"

0.5.4 gave the certificate a verdict on the phases that are **present**: does a
mixing phase want to come apart into two compositions? A verdict alone turned out
not to be actionable. A caller that means to act on it — by giving the phase a
second instance and starting it in the other lobe — needs the composition the
phase wants to move to, and that composition cannot be recovered afterwards: it
is a stationary point of the tangent-plane distance **in the full system**, fixed
jointly with the solution the phase sits in, and not a property of the mixing
model alone. It was computed and thrown away.

This release hands it back, adds a way to search for it where the default search
cannot reach, and exposes the linear program that decides whether an element
budget is feasible at all.

### Added — the incipient composition, alongside the verdict

`kkt_certificate` gains `split_trials`: per flagged phase, the member indices and
the mole fractions it wants to split into. `phase_tangent_trial` and
`phase_split_trial` are the underlying functions and are now exported; each
returns `(measure, trial)`, and `phase_tangent_measure` / `phase_split_measure`
are defined as `first` of them, so the two cannot drift apart.

### Added — `split_starts`, for a lobe the corners cannot reach

`SolutionPhase` gains a `split_starts` field (keyword, empty by default): extra
trial compositions for the split search.

The search starts from the corners of the composition simplex and refines by
successive substitution, which converges to the stationary point **nearest its
start**. A corner is usually on the right side of the barrier and sometimes is
not, and where it is not the iteration walks back to the phase's own composition
and reports nothing. Measured, on the published AFm sulfate/hydroxide binary of
CEMDATA18 (Redlich-Kister, A₀ = 0.188, A₁ = 2.49 in RT units; spinodal
[0.631, 0.914], binodal [0.4999, 0.9700]):

| phase sits at | corners | with the binodal as an extra start |
|:--|--:|--:|
| x = 0.5268 | +3.99e-02, trial x = 0.974 | same verdict |
| x = 0.95 | +2.4e-16, trial x = 0.950 — **missed** | +1.23e-01, trial x = 0.444 |
| x = 0.98 (outside the binodal) | stable | stable |

Both flagged compositions are metastable — inside the binodal, outside the
spinodal — so *being* metastable is not what defeats the corners; sitting in the
lobe they lead back into is. That is not something the solver can know in
advance, which is why this is an escape hatch a caller fills rather than a rule
the solver applies. Extra starts can only raise the maximum returned, so they
never take a verdict away, and the last row above is the test that they do not
invent one either.

The division of labor is deliberate: this package knows the system and not the
mixing model; the caller knows the model and can compute its binodal in
microseconds.

### Added — `simplex_start`, and an honest account of what it is worth

The initial approximation of GEM-Selektor, after Karpov: minimize the linear part
of the Gibbs energy over `A x = b, x ≥ 0`, whose optimum is a vertex with at most
`m` nonzero species. Two-phase simplex on a dense tableau with **Bland's rule**,
which cannot cycle — the polytope of a chemical system is massively degenerate,
and a cycling start routine would be worse than none.

It was transplanted because it is what a mature code does, and then measured. On
a 91-species Portland cement with 12 components:

| | |
|:--|--:|
| the LP itself | 0.27 s, `‖A x − b‖∞ = 6e-17`, 6 nonzeros of 91 |
| cold start, the caller's own cascade | 16.1 s, certified |
| the same cascade started from this vertex | 14.4 s, certified |
| this vertex with the cascade declined | 10.5 s, **not** certified |

Eleven percent, not the factor of eighty that separates a cold start from a warm
one — and the reason is structural rather than accidental. A vertex puts every
other species at the floor, which is the configuration a log-domain method
handles worst: the barrier gradients span the full range between a mole and the
floor, and a mixing phase's `ln x → −∞` on the very face where it vanishes.
GEM-Selektor starts there because its IPM is built around it; this solver's dual
Newton is not, and continuation on the loading is what works for it instead.

What the routine is genuinely worth here is the other half of its answer.
`nothing` is a **proof that the element balance has no nonnegative solution at
all** — a statement about `b`, which no iterative failure can establish. That
separates "this budget is impossible" from "the solver did not converge", and
those are different problems with different fixes.

### Breaking changes

- **`SolutionPhase` has a new field.** `split_starts` is appended, so code that
  constructs the struct by listing all its fields positionally must be updated.
  The documented keyword constructor is unchanged and gains an optional
  `split_starts`.
- **Below 1.0 the registry treats a minor bump as breaking**, whatever the API
  did. A downstream package pinned to `OptimaSolver = "0.5"` will not resolve
  `0.6` and must widen its bound to `"0.5, 0.6"` or `"0.6"`.

Nothing else changes: no answer moves, every existing signature still resolves,
and `kkt_certificate` only gains a field.

## v0.5.5 — the inner iteration ran 200 sweeps to move nothing

A performance release with a correctness-shaped cause, and the measurement is the
whole story. On a 90-species cement equilibrium, in a warm process so that no
compilation is counted:

| | 0.5.4 | 0.5.5 |
|:--|--:|--:|
| warm restart, bare dual solve | 583 ms | **18.5 ms** |
| fresh state, certified route | 689 ms | **49.2 ms** |
| against Reaktoro, same paste and database, same protocol | 21 × | **1.8 ×** |
| a coupled hydration trajectory | 594 s | 284 s (with the caller's own fixes) |

No answer changes. What changes is how long the solver spends not moving.

### Fixed — the inner convergence test measured the wrong thing

`_invert_phases!` recovers the phase compositions by successive substitution and
stops when its worst measure falls below a tolerance. That measure was a
**stationarity residual** for the aqueous phase. It should be, and now is, the
**displacement of the state** — which is exactly what the surrounding contract
asks for: `inner_tol` exists so that re-running the loop from its own output
changes nothing, and that is a statement about how far the state moves, not about
a residual.

The two differ precisely where it matters. `W` is a log-molality clamped to
`[-700, 20]`, and a species whose stationarity asks for less than the floor can
never reach it: the update is clamped, the state does not move, and the residual
stays where it is for ever. Profiling a warm solve put **98.4 %** of it in this
loop; counting showed 309 calls, 200 sweeps each, **100 % of them hitting the
cap**, with the worst measure at 1.0054e+02 on sweep 1 and 1.0054e+02 on sweep
200. It belonged to dissolved oxygen at 9.9e-305 mol — a reducing cement pore
solution puts O₂ far below the floor.

Measuring the displacement needs no special case for such a species: a clamped
update moves nothing, so it contributes exactly zero. It is also why the fix is
this rather than *excluding* those species from the measure, which was tried and
is wrong — a species sits at the floor only while the others hold it there, and
excluding it lets the loop stop one sweep before it comes off. That variant broke
a 25 % fly-ash paste whose element balance went from 5.8e-15 to 2.1e-01.

### What this re-enables, and why it deserves its own release

`DualNewtonOptions.inner_tol` gates the line search: a step is preferred when its
inner solve converged, so that the outer residual is a function of `v` alone
rather than of the warm start. Because the measure was pinned at 1.0054e+02, that
condition was **never true**, and the strict pass had never once fired on a
cement problem. It does now. That is a real change in how the solver chooses its
steps, and it is why this is a release rather than a line in another one.

Test suite: 285/285.

## v0.5.4 — a sentinel potential was being read as a chemical potential

Two things, and the first is a **correctness regression in 0.5.3** that anyone
on that release should move off.

### Fixed — the split test must not probe a composition the element balance forbids

`phase_split_measure`, added in 0.5.3, starts Michelsen's successive
substitution from each corner of the phase's composition simplex. It did so for
**every** end-member, including one whose conservation row is degenerate — no
matter of that component exists in the system, so the row's multiplier is pinned
at the sentinel `DEGENERATE_POTENTIAL` rather than solved for.

`uᵢ` for such an end-member is therefore not a chemical potential, and `uᵢ - gᵢ`
is a number with no meaning. Read by the measure, it reported a converged,
mass-balanced equilibrium as a phase wanting to move to a composition that
cannot exist.

Measured on a CEM III/A blastfurnace cement: `CSHQ` declared with its alkali
end-members `KSiOH` and `NaSiOH`, on a paste whose clinker and slag bring no
potassium and no sodium. Under 0.5.3 the certificate returned

| | 0.5.1 | 0.5.3 | 0.5.4 |
|:--|--:|--:|--:|
| element balance | 3.8e-14 | 3.8e-14 | 3.8e-14 |
| pH, assemblage | identical | identical | identical |
| `worst_violation` | negative | **+54.06** | negative |
| `optimal` | `true` | **`false`** | `true` |

The composition never changed — only the verdict did. And **+54.06 was the same
number on every unrelated system carrying the same declaration**, which is the
signature of a sentinel rather than of chemistry.

`phase_tangent_measure` and `phase_split_measure` now take the `dead` set and
exclude those members from the trial composition and from the log-sum-exp, which
is the guard `_mole_fraction_exponents` has carried from the start; the two
call sites in `kkt_certificate` pass it. A phase with fewer than two live members
has no interior to unmix into and returns `-Inf`.

This does not weaken the test it was added for: a genuine miscibility gap has
both of its lobes made of end-members the element balance can supply.

### Fixed — the gradient configuration was rebuilt on every iteration

A pure performance change: no API change, no different answer — the same solve,
62 times cheaper in the term that dominated it.

### Fixed — `ForwardDiff.GradientConfig` is built once per element type

The interior point's gradient fallback read

```julia
(grad, u, par) -> ForwardDiff.gradient!(grad, v -> f_obj(v, par), u)
```

with no configuration passed, so ForwardDiff rebuilt the whole `GradientConfig`
— the dual seeds and every work buffer — on each call. It is called once per
iteration.

Measured on the Jacobian of that closure, for a 90-species cement paste:

| | cost | relative to one function call |
|:--|--:|--:|
| configuration rebuilt each call | 22.47 ms | 3295 × |
| configuration built once | 0.36 ms | 53 × |

against an ideal of roughly 24 × for 90 inputs at chunk size 12. This was the
dominant term of a cold cement solve, which is also why a starting point nearer
the answer had barely moved it: the cost was never in the starting point.

The configuration is kept **per element type** of `u` rather than cached blindly.
A solve differentiated with respect to its parameters arrives here with `u`
seeded as `ForwardDiff.Dual`, and a configuration built for `Float64` cannot
serve it. One per type reuses inside the iteration and rebuilds only when the
type changes, so the package stays differentiable end to end.

Test suite: 274/274, unchanged.

## v0.5.3 — a phase that is present can still be the wrong answer

The optimality certificate tested a mixing phase held **absent** with Michelsen's
tangent-plane measure, and tested one that is **present** by the stationarity of
its members and nothing else. Stationarity is blind to the single failure that
matters for a non-ideal phase: that the Gibbs minimum for it is *two* coexisting
compositions rather than the one reported.

Convexity is what made that safe to ignore. A Redlich-Kister excess term strong
enough to open a **miscibility gap** is exactly where convexity fails, and it is
not a hypothetical case — the published parameters of the AFm sulfate/hydroxide
and AFt sulfate/carbonate binaries of CEMDATA18 are concave over an interval.

### Added — `phase_split_measure`

The largest tangent-plane distance reachable from any start **other than the
phase's own composition**, and it needs its own function rather than a keyword on
`phase_tangent_measure` for a reason worth recording: at equilibrium the members
of a present phase satisfy `uᵢ = gᵢ + hᵢ`, so the tangent-plane distance at that
composition is exactly zero *and is a stationary point of the successive
substitution*. Started uniformly — which is the right start for an absent phase —
the search walks straight to it and reports zero however unstable the phase is.
The starts used here are the end-member corners, which lie in the other lobe when
there is one.

`kkt_certificate` now folds that measure into `worst_violation` for every present
mole-fraction phase, and reports `worst_violation_split` and `split_phases`
alongside the existing fields.

### What it does not touch

Nothing on a convex system: there the measure is zero at equilibrium by
construction. The full suite is unchanged at 268/268.

One restriction is deliberate and was found by that suite rather than by
reasoning. The measure applies to **mole-fraction phases only**. The aqueous
solution is a phase in this formulation too, but its activities are molalities
referred to the solvent rather than mole fractions of its own members, so the
measure — which is written in the latter — computes nothing meaningful for it.
Applied indiscriminately it made two convex reference problems fail to certify.
It also cannot unmix: there is one solvent.

### Breaking changes

Nothing in the API breaks and no default behavior changes on a convex problem.
Two consequences are breaking in practice:

- **A non-convex mixing phase that used to certify now may not.** That is the
  point of the release: such an answer was a KKT point and not a minimum, and the
  certificate said otherwise. Code relying on the old verdict was relying on a
  claim the solver could not support.
- **The registry treats a minor bump below 1.0 as breaking whatever the API
  did**, so a downstream bound pinned to `"0.5"` accepts this patch release, but
  `ChemistryLab` must require `OptimaSolver = "0.5.3"` to get the check at all.


## v0.5.2 — a step whose inner solve has not converged is not a descent step

The outer residual of the dual Newton is **not a function of `v` alone**.
`_outer_residual` calls `_invert_phases!`, which recovers the composition of each
mixing phase by a damped fixed point started from the `W` it is handed and
mutated in place. Until that fixed point has converged its answer depends on the
warm start, and so does the residual computed from it.

The backtracking line search compared the two anyway: it evaluated a candidate
from a frozen copy of `W`, accepted on a decrease, and wrote the new `W` back;
the next iteration then recomputed the residual from that new `W` and could get a
different number. Measured on a cement with eight solid solutions, a step
accepted as decreasing from 60.5 was followed by a residual of 4.2e17, and the
active-set round recorded that 4.2e17 as the state's KKT error.

### Bug fixes

- `_invert_phases!` reports the residual its sweeps ended on, so a converged
  inversion can be told from one that merely ran out of sweeps.
- The inner Newton loop accepts a step in **two passes**: first a step that both
  decreases the residual and whose inner solve converged, then — only if none was
  found — a step that merely decreases it. The second pass is not a concession:
  an inner iteration whose `h` does not depend on the composition has nothing to
  solve and can never report convergence, and refusing every step there would
  stall the solve at its starting point.

### Added

- `DualNewtonOptions` gains `inner_tol` (default `1e-10`) and `inner_maxit`
  (default `200`), naming what was a hard-coded threshold and sweep cap.

No exported name or signature changes; 268 of 268 tests pass.

## v0.5.1 — three defects the SciML entry point was hiding

The documented way into this package is the SciML one: hand an
`OptimizationProblem` to `OptimaOptimizer()` and never see `OptimaProblem`.
Nothing in the test suite went that way — every test called the internal
`solve` — so `sciml_interface.jl` sat at **0 % coverage** while being the file
every downstream caller actually runs. Covering it turned up three real
defects, all fixed here. No exported name or signature changes.

### Bug fixes

- **The constraint matrix was recovered with a step size that made it wrong in
  the ninth digit.** `_extract_constraints` rebuilds `A` by differencing the
  residual. For the affine residual the interface documents, the quotient is
  exact whatever the step, so the only error left is cancellation — and the
  reflex `ε = 1e-7` maximizes it: against residuals of order one, `A` came back
  with a relative error near `1e-9`. That is a floor the solver cannot get
  below, so a `tol = 1e-12` run stalled at `3e-12` and reported `MaxIters` on a
  problem it had solved. The step is now the scale of the variable, which
  brings `A` back to within an ulp.

  Callers do pass nonlinear residuals all the same — a log-parameterized
  equilibrium sends `A exp(x) - b`, where a step of one is 72 % off the
  derivative — so affinity is now **verified per column** rather than assumed:
  each column is differenced at two scales and the large answer is kept only
  when the two agree. A nonlinear residual keeps the local Jacobian it always
  got.

- **`_project_alternating!` could return a point with negative amounts.** It
  measured the affine residual *before* projecting onto the bounds, so a point
  already satisfying `A n = b` with entries below `lb` was returned untouched —
  and then reached the barrier, where `log` of a non-positive amount is `NaN`,
  with nothing in the trace pointing back. The box projection now comes first.

- **A problem with no equality constraint died on a `BoundsError`.**
  `Canonicalizer` pivots a QR of `A`; for a zero-row matrix LAPACK returns a
  degenerate permutation and the failure surfaced several calls later.
  `OptimaProblem` now rejects it with a message saying what the solver needs,
  and the SciML path says the same when neither `cons` nor `A`/`b` is supplied.

### Testing

Coverage goes from 74 % to 93 %, with `newton_step.jl`, `canonicalizer.jl`,
`convergence.jl`, `sciml_interface.jl`, `stability.jl`, `residual.jl` and
`sensitivity.jl` fully covered. Beyond the SciML path, what is now exercised
and was not: the `solve` overload taking a pre-built `Canonicalizer`, the
barrier-exhaustion exit, the Lawson-Hanson NNLS active-set release, the
variable-stability classification, the line-search filter, `refactorize`,
`binding_variable`, the rank-deficient conservation matrix, an infeasible
budget, the verbose logging, and the inertia correction — including the
Bunch-Kaufman fallback taken when no ridge below the ceiling restores
definiteness.


## v0.5.0 — a feasibility error that could only see the largest budget

### Breaking changes

- `error_feas` is now scaled row by row, so `tol` means a **relative** accuracy
  per conservation row. Code comparing it against an absolute mole tolerance must
  be revisited; the unscaled figure is still available as `error_feas_abs`.
- `KKTResidual`, `OptimaState` and `OptimaResult` each gain a field
  (`error_feas_abs`). Positional constructors need the extra argument.
- `DualNewtonProblem` gains six keywords (`gq`, `hq`, `cq`, `q0`, `qscale`, `Aq`,
  `always_active`) and two type parameters. Constructed by keyword, as documented,
  nothing changes.
- `kkt_certificate` returns three more fields (`worst_violation_bounded`,
  `worst_violation_phase`, `absent_phases`) and `worst_violation` now includes the
  mixing-phase test, so a composition that omits a stable solid solution is no
  longer certified.
- Downstream packages must widen their bound to `OptimaSolver = "0.5"`.

### A block of unknown parameters, so a prescribed property is not an outer loop

`DualNewtonProblem` takes `q0` unknowns with `cq` residuals, `gq` for a standard
part that depends on them, `hq` for an activity model that does, and `Aq` so a
linear row may involve them. The outer Newton system grows by `nq` equations and
`nq` unknowns, and that is the whole cost.

It is the vehicle for everything a closed system at fixed `(T, P)` cannot express:
a prescribed enthalpy with the temperature unknown, a prescribed volume with the
pressure unknown, a prescribed activity with a titrant amount unknown, and the
reaction extents of an implicit kinetic step. `ChemistryLab` 0.14.0 uses all four.

`hq` is not a convenience. The Debye-Hückel coefficients are functions of
temperature, so an adiabatic solve whose activity model stayed at the starting
temperature would minimize a Gibbs energy that does not exist. A test shows the
same problem without `hq` failing to converge.

`Aq` carries the structural trick of Leal's kinetics: the reactivity constraint
`Kᵀn − Δξ = ξ₀` is **linear**, so it belongs in the conservation block beside the
elements and the charge, and the algebraic cost of kinetics is the number of
reactions rather than of species.

`always_active` marks bounded variables pinned by a linear row. A variable whose
amount is fixed by a constraint is not deciding anything by a sign test: it is
determined, and the row's own multiplier makes its stationarity satisfiable at
whatever amount is demanded. Without it such a variable can never enter, because
it starts at zero and the drop rule removes anything below `si_tol` — which is
what happened to a mineral exhausted at equilibrium, leaving its reactivity row
unenforced.

### The degeneracy criterion was asked about rows it does not describe

`degenerate_components` answers a question about **element conservation**: a row
whose non-zero entries share a sign and whose budget is zero forces every variable
in it to vanish. It was applied to every row of `A`, and once `A` carries rows
that mean something else the answer is wrong.

The row that provoked this is a reactivity constraint,
`nᵢ − Σⱼ νᵢⱼ Δξⱼ = nᵢ(0)`: one positive entry on `x` and, for a product that
starts absent, a zero right-hand side — exactly the shape the criterion reads as
"this component is absent from the system". It then pinned that row's multiplier
to `DEGENERATE_POTENTIAL` and declared the species dead, so a solid product could
never form. Measured downstream, the stationarity residual sat at 458 and the
reaction extents came out 4.3 times short.

`conservation_rows` says which rows may be asked, and defaults to all of them, so
a problem that declares nothing behaves exactly as before.

### The certificate's stationarity is scaled; its feasibility deliberately is not

The entries of `∇f` are chemical potentials referred to the elements, of order
10²–10³ in `RT` units, so an absolute threshold of 1e-10 on their residual asks
for thirteen digits of cancellation. On a system whose multipliers are that
large — a kinetic step pinning a mineral by a linear row, whose multiplier must
reach that mineral's own potential — the answer was right to nine digits while the
certificate reported 2.9e-8 and refused it. The residual is now divided by the
size of the quantities it is built from, never below one, so a well-scaled problem
is judged exactly as before. `stationarity_abs` and `stationarity_scale` report
both halves.

The feasibility stays absolute, and that is a decision rather than an oversight.
The two measures answer different questions: the solver's schedule needs to know
whether a row is satisfied relative to its own budget, so that a small element is
not hidden behind a large one; the certificate states how much matter the
composition fails to account for, which is a number of moles. Scaling it was tried
and is wrong — the charge row of a dilute solution has a zero budget and a flux of
order 1e-6, so dividing by it turns 7.6e-7 mol of machine noise into 0.76 and
refuses every answer, the correct ones included.

### The certificate could not see a solid solution that should have formed

A mixing phase held **entirely** absent was examined by neither of the
certificate's two tests: its members are not interior (they sit at the floor) and
not bound-constrained (a phase member's condition is an equality, since
`ln xᵢ → −∞`). So a composition omitting a solid solution passed unexamined — the
one soundness hole the certificate had, and precisely the case that matters for a
cement, whose C-S-H is a mixing phase.

`phase_tangent_measure` closes it: Michelsen's tangent-plane measure with the
trial composition refined against the phase's **own** activity model by successive
substitution, so the test is not the ideal approximation. It enters
`worst_violation`.

While there, a misleading comment is corrected. `_phase_tangent` — the *search*
heuristic, which decides which candidate phase to try next — takes `lnγ` as zero
and always did, while its comment claimed the activity model was read. A
first-order test is the right cost for a search, and nothing in the proof depended
on it: `kkt_certificate` uses the true `∇f = g + h(x)`.

The certificate's feasibility residual now includes `Aq q`. Without it the
parameter itself is reported as an infeasibility: on a kinetic step the residual
came out at exactly `Δξ`.

### The rows of a conservation matrix do not share a scale

`error_feas` was `‖An − b‖∞`, on the argument that moles are absolute. True, and
beside the point: the rows of a chemical conservation matrix carry budgets that
differ by orders of magnitude, so one norm reports the largest and hides the rest.

Measured on 1 mmol of calcite dissolving in 1 kg of water, at the answer the
solver returns:

| row | budget | residual | relative |
|---|---|---|---|
| H | 111.0 mol | 1e-6 | 9e-9 |
| O | 55.5 mol | 8e-6 | 1.4e-7 |
| Ca | 1.0e-3 mol | 1e-6 | 1e-3 |
| CO₃ | 1.0e-3 mol | 3e-6 | 3e-3 |
| charge | 1.0e-4 mol | 3e-6 | **3e-2** |

`‖An − b‖∞ = 3e-6` looks like round-off and is reported as such. The charge
balance of the composition handed back is wrong in the second digit.

Each row is now divided by `max(|bₖ|, Σⱼ |Aₖⱼ| nⱼ)` — see `row_scales`. The flux
term is what makes a zero budget meaningful: a charge row sums to zero by
construction, so no relative error exists against `b`, but the ions carrying it
are of size 1e-4 and that is the scale to judge against.

The change does not repair the iteration, and was not expected to: on the same
case the solve still ends on `MaxIters`, now for the right reason. What it does is
stop the error from being invisible. The accuracy itself is a `ChemistryLab`
question, answered there by solving through both back ends and keeping the answer
the KKT certificate proves optimal — which certifies ten of ten reference cases
where the interior point alone certifies seven.

### Why the interior point cannot close that residual

Traced on the same case: from iteration 104 to 127 the error sits at exactly
3.0e-6, `α` is pinned at its ceiling of 0.15, and `‖dn‖` falls geometrically at
`1 − α` per step. The Newton direction is not reducing the residual; the
fraction-to-boundary rule lets through 15 % of a correction the next iteration
re-poses. Raising `max_iter` to 30 000 changes nothing, because the outer loop
exits on `μ ≤ barrier_min` rather than on the iteration count, and loosening
`tol` to 1e-6 changes nothing either.

This is the regime the dual Newton was written for: its interior variables are
parameterized by `w = ln x`, so no fraction-to-boundary rule applies to them, and
only the bounded variables carry an active set. It is 3e-12 on the same case.


## v0.4.3 — an abstractly typed parameter is not a working precision

Fixes a regression introduced in v0.4.2. **If you are on v0.4.2, upgrade.**

### `param_eltype` promoted to `Real`, and the solver died on `eps(Real)`

v0.4.2 added `p` to the promotion that fixes an `OptimaProblem`'s element type, so
that a parameter seeded with `ForwardDiff.Dual` numbers reaches every workspace.
It promoted to whatever `eltype` the parameter's containers reported — including
an **abstract** one.

That is not a hypothetical. `ChemistryLab`'s SciML path passes its stoichiometric
data inside `p` as `A::Matrix{Real}` and `b::Vector{Any}`, because that is how its
matrices are built. `eltype` of those is `Real` and `Any`, which name no
arithmetic: the promotion produced `OptimaProblem{Real}`, `solve!` asked for
`eps(Real)`, and there is no such method. Every equilibrium solve through that
interface failed — `ChemistryLab`'s own test suite went red on v0.4.2.

`param_eltype` now contributes a type only when it is **concrete** and a
`Number`. That is precisely the case the promotion exists for, since a
`ForwardDiff.Dual` is always concretely typed, and it ignores the abstract
containers that carry no precision. A parameter mixing both — the real
`ChemistryLab` case, `Float64` scalars beside a `Matrix{Real}` — contributes
`Float64` from its concrete leaves alone.

### What the v0.4.2 tests missed, and what now covers it

v0.4.2 shipped a regression test for the case it fixed: differentiating a full
solve with respect to `μ⁰`. It passed, and it still does. What no test exercised
was a parameter carrying an abstractly typed container, which is the shape every
real caller of the SciML interface uses. The suite now builds a problem with
`p` holding `Matrix{Real}` and `Vector{Any}`, asserts the element type stays
`Float64`, and solves it.

## v0.4.2 — differentiating with respect to a parameter

No API change and no numerical change to any answer that already computed: the
same compositions, the same certificates, the same element balances.

### `OptimaProblem` ignored the element type of `p`, so AD through a parameter failed

`OptimaProblem` inferred its element type as

    T = promote_type(eltype(A), eltype(b), eltype(lb), eltype(ub))

`p` is typed `Any` and was left out. Seeding it with `ForwardDiff.Dual` numbers —
differentiating an equilibrium with respect to standard chemical potentials, say —
therefore left `T` at `Float64`, so `solve!` allocated its gradient buffer as a
`Vector{Float64}` and the caller's `g!` died on the first write:

    MethodError: no method matching Float64(::ForwardDiff.Dual{...})

What made this hard to notice is that everything around it worked. Differentiating
with respect to `b` was fine, because `eltype(b)` *is* in the promotion, and the
test suite exercised exactly that. Every building block — `gibbs_hessian_diag`,
`kkt_residual`, `hessian_diagonal`, the objective — was separately verified to be
Dual-compatible. Only a full solve seeded through `p` was not, and that is the case
the documentation claimed worked: *"Because Optima uses generic arithmetic
throughout, ForwardDiff can differentiate the entire sensitivity computation with
respect to parameters"*.

`param_eltype` now contributes the numeric element type carried by `p`, descending
into tuples and named tuples, and returning `Union{}` — neutral for
`promote_type` — for anything it does not recognize as numeric. A parameterless
problem is unaffected.

The regression test differentiates a full solve with respect to `μ⁰` and compares
against the analytic sensitivity `∂n_∂μ0`, which never sees a dual: the two routes
to the same Jacobian, implicit function theorem and forward AD, now agree to
`7e-12`.

### The documented examples are executed, and four of them did not run

`docs/src/examples/` held twenty-three `julia` blocks that Documenter never
executed, so nothing checked them. They are now `@example` blocks, and making
them run required fixing what they said:

- `basic_usage.md` and `warm_start.md` called `μ⁰_at_temperature`, a function
  defined nowhere in the package or the documentation. Both now define an
  explicit placeholder, labeled as such — the subject of those sections is
  reusing a `Canonicalizer` and warm-starting, not thermodynamics.
- `sciml_interface.md` built its central `SciMLBase.OptimizationProblem` with
  `lb` and no `ub`, which SciMLBase rejects outright: *"If any of `lb` or `ub` is
  provided, both must be provided."* The page's main example had never run.
- `basic_usage.md` set bounds on a problem built from `f, g!`, names that do not
  exist on that page, and omitted `p` — so the objective would have failed on
  `p.μ⁰`.
- The "Sample output" transcription of the verbose iteration log had neither the
  right columns nor the right prefix. The executed block prints the real log, so
  the transcription is gone.

Two blocks remain illustrative on purpose and say so: the `ChemistryLab` usage
example, because `ChemistryLab` depends on `OptimaSolver` and not the reverse, and
the warm-start cache sequence, which refers to problems built elsewhere.

`docs/Project.toml` gained `[sources] OptimaSolver = {path = ".."}`. Continuous
integration already ran `Pkg.develop(path=pwd())`, but a local documentation build
without it resolved the *registered* version — documenting a release rather than
the working tree.

## v0.4.1 — the barrier floor no longer costs 285 empty iterations

No API change, and no numerical change to any answer: the same compositions, the
same certificates, the same element balances.

### `reduce_barrier` clamps at `barrier_min`, so the exit test never fired

The outer loop ended on `μ < barrier_min`, and `reduce_barrier` returns
`max(barrier_min, decay·μ)` — a value that is never below the floor. Once `μ`
reached it, the loop therefore re-entered the inner iteration up to `max_iter`
times, each round re-evaluating the same residual and breaking again on
`should_reduce_barrier` **without taking a single step**, so not one of those
iterations could help. The test now asks whether the barrier was already at its
floor, which is the condition that means nothing further is available.

Measured: a warm-started three-species solve reported 312 iterations for work that
had finished at 27, and cement equilibria that stop short of the requested
tolerance — which is most of them, the tolerance being tighter than a difference of
chemical potentials can be resolved to in Float64 — fell from 0.17–0.21 s to
0.03–0.07 s each. In a coupled kinetics run there is one such solve per accepted
step plus the Jacobian probes, so it compounds.

### One assertion corrected because it measured the wrong thing

`test_solver.jl` asserted that a warm-started solve costs fewer iterations than the
cold solve of a **different** problem, at `tol = 1e-12`. Neither half of that is a
test of warm starting: two problems need not cost the same, and twelve digits of an
objective of size 0.6 puts both solves in the regime where the Armijo test is
settled by rounding, where the iteration count is not even reproducible across
environments — Julia 1.13.0-rc3 under coverage instrumentation reported 46 where a
plain 1.12 and 1.13 both report 55, and the assertion failed at `52 < 51` while the
warm start was working.

The control is now the same problem solved cold, at `tol = 1e-10`. Measured
identically on 1.12 and 1.13: 21 iterations warm against 26 cold, with the answer
right to 4.9e-10; the saving is a consistent four to five iterations at 1e-6, 1e-8
and 1e-10, and only at 1e-12 does it invert.

### A complete primal-dual port, measured, and deliberately not adopted

The one substantive difference between this primal barrier method and the
primal-dual method Optima and Ipopt implement is the curvature term: `Σᵢ = zᵢ/sᵢ`
with an iterated bound multiplier, against `Σᵢ = μ/sᵢ²`. It was implemented in full
— the `z` iterate, its Newton step `δz = μ/s − z − Σ δn`, its own
fraction-to-boundary step length, the `κ_Σ` safeguard of Wächter & Biegler (2006)
Eq. (16) — and then removed, because it is measurably worse on the problems this
package exists for. The finding is recorded at `hessian_diagonal` so it is not
re-attempted blindly.

The reason is structural. A pure phase's Gibbs energy is linear in its amount, so
`∂²f/∂nᵢ² = 0` exactly and `Σᵢ` is not a correction to the curvature — it is the
whole curvature in that direction. `μ/sᵢ²` is then the exact Hessian of the barrier
subproblem being solved, and Newton's method on it is exact; `zᵢ/sᵢ` replaces it
with a quantity that only tracks the central path approximately. A primal-dual
method earns its keep where `∇²f` supplies the curvature and the multipliers carry
information the barrier does not, which is the opposite regime.

| | `Σ = μ/s²` | `Σ = z/s` |
|---|---|---|
| LC³ clay sweep, five replacement levels | **5/5 certified** | 4/5, fails at 30 % |
| stationarity at LC³-50 | 2.9e-11 | 2.1e-10 |
| coupled calcite trajectory | 1 solve short of tolerance | identical |
| three-species ideal solution | — | identical to the last digit |

Two further errors of that port are recorded at `kkt_residual`, both about what
convergence may be judged on: measuring `‖∇f + Aᵀy − z‖∞` reported twelve solves
short of tolerance where the primal barrier reported one, the answers being
identical, because a raw difference of chemical potentials of order 10²–10⁵ RT
cannot cancel to 1e-10 in Float64; and replacing complementarity by `max|sᵢzᵢ|` to
compensate is vacuous, since `zᵢ → 0` wherever a variable is interior.

### One noise-dominated regime, diagnosed and left alone

Below the resolution of the objective the Armijo test is settled by rounding rather
than by the function, and backtracking then halves the step at random: on the
three-species solution, seventeen consecutive iterations returned `α` of 0.0039,
0.0078 and 1.91e-6 while `‖dn‖` sat at 4e-12. Two remedies were implemented and
both measured worse — taking the full step gained 55 → 32 iterations on the toy
problem but moved the Reaktoro coupling reference's `t = 0` pure-water speciation
from 0.9 % to 18.8 % wrong on OH⁻, and treating the condition as "this barrier
level is finished" is not a proximity test at all and returned an answer wrong at
1.3e-6. Ipopt's own tiny-step check does nothing in this regime either. The
diagnosis is recorded in `line_search`; the code is unchanged.

## v0.4.0 — mixing phases that cannot run away, and an active set that exchanges

### Breaking changes

Two new exported names, `SolutionPhase` and `stationarity_capacity`, and a new
keyword `mole_fraction` on the first.
Nothing was removed or renamed and no existing signature changed, but below 1.0
the resolver treats a minor bump as breaking regardless, so a downstream
`[compat] OptimaSolver = "0.3"` must be widened to `"0.4"`.

### A solid solution could not be solved at all, and said so by diverging

`dual_newton_solve` recovered every non-reference member of a mixing phase from
its own stationarity, `hᵢ = uᵢ − gᵢ`. That is right for an aqueous solution,
where the solutes carry molalities and `hᵢ` is unbounded above. It is impossible
for a solid solution: there every member is a mole fraction, every `hᵢ` is
bounded above by zero, and a positive right-hand side simply cannot be met. The
iteration answered by growing the member without bound and stopped only at the
internal clamp — `exp(20) = 4.85e8` mol of C-S-H, with the outer Jacobian then
computed on that.

Fixing the reference's amount and inverting the rest does not repair it either:
the phase total works out to `x_ref / (1 − S)` with `S = Σ_{i≠ref} exp(uᵢ − gᵢ)`,
which has no positive solution once `S ≥ 1` — exactly when the phase is
supersaturated at the current multipliers.

What the potentials do determine, for any `y`, is the *composition*. A phase
declared `mole_fraction = true` is now recovered as
`xᵢ = N · softmax(uᵢ − gᵢ − ln γᵢ)` with the phase total `N` as the outer
unknown, and its outer equation is `logsumexp(uᵢ − gᵢ − ln γᵢ) = 0`. Both are
evaluated with the maximum factored out, so nothing overflows. The phase equation
is now the same expression as the tangent-plane admission test, so a phase is
admitted and held stationary by one quantity rather than two that could disagree.

### The active-set search had no leaving rule, and no way out of a wrong set

Three moves were missing, and without them the search could not reach the answer
on a system with many candidate phases. Each is a KKT condition, not a tuning
choice.

**Gibbs' phase rule was not enforced.** A bound-constrained variable held active
imposes `uᵢ = gᵢ`, i.e. `aᵢᵀ y = −gᵢ`, one linear equation in `y`; a mole-fraction
mixing phase imposes `logsumexp(uᵢ − gᵢ) = 0`, one more. With `y ∈ ℝᵐ` no `y`
satisfies more than `m` of them, and the composition vectors of the active
variables must in addition be linearly INDEPENDENT — two dependent columns demand a
fixed relation between their `gᵢ` that no database satisfies, which is what two
polymorphs of one composition amount to. An active set breaking either condition
cannot support a solution: the residual cannot reach zero for any iterate and the
least-squares step merely spreads the violation. On an LC³ equilibrium the set grew
to 15 pure phases and 5 solid solutions — **19 conditions on 12 components** — and
the solve returned a stationarity residual of 18 having never had a solution to
find. Both conditions are now invariants of the active set, maintained by exchange:
`stationarity_capacity` and the rank test decide, and admitting a violated
candidate releases the incumbent that makes room.

**Complementarity was tested on one side only.** The conditions for a
bound-constrained variable are `xᵢ ≥ 0`, `sᵢ ≤ 0`, `xᵢ sᵢ = 0`. The drop test read
`xᵢ → 0` and nothing else, so a variable held active while UNDERSATURATED — `sᵢ`
strictly negative — stayed for ever, and no Newton iteration could repair it
because its own equation `sᵢ = 0` is the one that cannot hold. Measured on an LC³
equilibrium at a quarter of full reaction, the search settled with nothing
supersaturated, the element balance at 4.5e-2, and a stationarity residual of 9.86
carried entirely by such a phase. The test now covers both halves, evaluated on a
point that actually solves the current subproblem — while the inner Newton is still
working, `sᵢ` is a transient and not a violation.

**The search was not a descent method.** Each move is a guess, and a guess that
makes things worse has to be undone rather than built upon; there was no such
mechanism, and the search wandered — 16.04 → 8.05 → 16.04 → 514 on an LC³ budget,
passing through and abandoning its best state. It now keeps the best state and
returns that, and the measure it descends is the KKT error of the WHOLE problem,
not the residual of the subproblem the current set defines. That distinction
decides the search: a set omitting a phase the solution needs solves its own
equations exactly — residual 1e-12 — while the omitted phase sits absent and
supersaturated by ten RT, so ranking on the subproblem residual rewards leaving
phases out.

**A phase whose every member carries a vanished component was seeded as present.**
Its stationarity condition is evaluated over an empty set: every exponent is
`-Inf`, the log-sum-exp is `-Inf`, and the outer residual is `Inf` from the first
evaluation. The admission test already excluded such a phase; the seeding did not,
and against an interior-point warm start that matters, because a barrier point
holds even a dead species near `μ` rather than at zero. On an LC³ budget, which
carries no magnesium at all, that seeded the M-S-H or hydrotalcite solution as
present and no finite residual was ever produced.

**The seeding itself read a threshold that means nothing after a barrier solve.**
`n0[i] > 1e-6` was the test for "the guess holds this phase", and a barrier point at
`μ` holds every ABSENT phase at `sᵢ = μ/gᵢ` — for `μ = 1e-6`, exactly the threshold.
Candidates are now taken in order of decreasing amount, since the phase rule says
at most `m` are present and the abundant ones are the best guess as to which, and
admitted only while the set still supports a solution.

Measured together: the LC³ sweep of the private low-carbon work — a limestone
calcined clay cement from 0 % to 40 % clay replacement — is **certified optimal at
every point**, with stationarity 1.1e-11 and element balance 5.0e-11 at LC³-50, no
phase supersaturated, and a pore solution at pH 12.71. It previously returned a
stationarity residual near 20 with an element balance of several hundred moles.

### The initial multipliers came from fits that ignored the element budget

The three starting `y` were least-squares fits of `Aᵀy ≈ −(g + h)` over subsets of
species. That minimizes a stationarity residual and says nothing whatever about
`b`, so the potentials it produces are consistent with no composition in
particular — and on a cold start that showed: at a quarter of full reaction on an
LC³ budget the inner Newton failed to converge on EVERY active set the search
visited, residuals between 8 and 19, while the same problem reached by continuation
from a nearby solution converged to 1e-11.

A fourth candidate is now solved for properly. Treating every species as an ideal
one whose activity is its own amount, the Lagrangian minimizes in closed form,
`xᵢ = exp(uᵢ − gᵢ)`, and the dual becomes

```
φ(y) = −bᵀy − Σᵢ exp(uᵢ − gᵢ),   ∇φ = A x − b,   ∇²φ = −A diag(x) Aᵀ ≺ 0 .
```

`φ` is smooth and **strictly concave**, so Newton with a backtracking line search
converges from any starting point — no active set, no combinatorics, nothing to
stall in. This is Brinkley's method, after White, Johnson & Dantzig (1958), and its
answer is the `y` for which the ideal composition conserves matter exactly. It is
the last candidate tried, not the first: a caller replaying a trajectory hands over
a composition that is nearly the answer, the fits built from it converge
immediately, and this solve is then never run. Ordered the other way it cost a
warm-started replay three digits of element balance for no gain.

The candidates are also **ranked by the KKT error they reach** rather than by the
order they were tried in. Keeping the first unless a later one *converges* discards
a better answer whenever none converges: a start landing at 1e8 was returned in
preference to one at 20, because neither had crossed the tolerance.

### The difference step was scaled to the unknown, not to the residual

The outer Jacobian is differenced, and the step was `1e-5·|v_k|`. For the
multipliers that is the wrong scale by five orders of magnitude: the `gᵢ` are Gibbs
energies of formation FROM THE ELEMENTS, of order 10²–10³ in RT units, `y` carries
that offset, and the residual depends on `y` only through `u = −Aᵀy` and
exponentially so. The step came out near `5e-3`, moving `u` by `|A|·5e-3 ≈ 0.15` and
every `exp(uᵢ − gᵢ)` by some sixteen percent — a secant across a wide interval, not
a derivative. The scale over which the residual varies with `y_k` is
`1/maxᵢ|A[k,i]|`, and that is what now sets the step.

### An active set that rejected the variable it should have exchanged

An admission that failed to converge was undone by rejecting the entrant
permanently. That reads the failure backwards. The inner Newton stops the moment
an active variable falls below its bound — that is the *departing* variable
announcing itself, and letting the ordinary drop path remove it while the entrant
stays is the exchange an active-set method is supposed to perform.

On a cement, ettringite and monosulphate compete for the same sulfate, so
admitting one necessarily drives the other out. Rejecting ettringite let the
solve converge — to `2e-12` stationarity and `5e-12` element balance — onto an
assemblage in which ettringite was **absent and supersaturated by 14.8**. Only
the certificate caught it, and nothing in the run said so.

The entrant is now reconsidered only when the Newton failed with *nothing*
leaving, which is the genuine over-determination the guard was written for. A
veto lasts only as long as the active set that produced it, and a vetoed variable
that is still supersaturated prevents the run from being reported as converged:
the candidate list is filtered, the KKT conditions are not.

### Convergence was judged at the current barrier level, not at the optimum

`is_converged` compared `max |sᵢ (∇f + Aᵀy)ᵢ − μ|` against `tol`. That quantity
vanishes at the solution of the barrier subproblem **whatever `μ` is**, so the
solver could report success on a point `O(μ)` away from the actual optimum — and
which barrier level it happened to stop at, hence whether the answer was accurate
to 1e-8 or to 1e-10, depended on the inner-loop schedule rather than on anything
the caller asked for.

The test is now Ipopt's `E_0` (Wächter & Biegler 2006, Algorithm A, step 2): the
same residual evaluated at `μ = 0`, which is the true KKT error and cannot be met
at a loose barrier. `KKTResidual` gains `error_0` alongside `error`; the barrier
schedule still uses the μ-dependent one, which is what it is for.

`barrier_decay` moves from 0.1 to Ipopt's `κ_μ = 0.2`, and the two changes belong
together: with the honest test and the aggressive schedule the barrier outruns the
inner Newton and the error plateaus just above the tolerance without crossing it —
312 iterations to reach 9.999e-11 against a tolerance of 1e-10. At 0.2 the same
problem converges in 30, and a three-species ideal system reaches its exact
Boltzmann distribution to 7.6e-12 where it previously stopped at 8.1e-8.

`barrier_eps_factor` is exposed for callers who scale their optimality error, and
**defaults to 1, not to Ipopt's `κ_ε = 10`**: `κ_ε` applies to their *scaled* `E_μ`,
so carrying it across to an unscaled error is not adopting their criterion but
loosening ours by an unjustified factor. Measured at 10, a warm-started replay of
a calcite trajectory came back with element-balance residuals of 4.7e-8 instead of
1.4e-11, every solve having stopped one barrier level short.

### The rank test and the basis order are two questions, and they had one answer

`Canonicalizer` chose both from one pivoted QR of `A` as handed to it, and the
SciML interface hands it `A · diag(s)` with `s` the starting value of each
variable. That single answer was wrong for each question in the opposite
direction.

For the RANK, the scaling is noise: `rank(A · diag(s)) = rank(A)` for any positive
`s`, but the pivoted-QR test compares each pivot to the largest, and warm-starting
from a converged equilibrium spreads the columns over ten orders of magnitude. The
rank came out one short, `B` was built with `m−1` columns, and the run died inside
LAPACK with "matrix is not square". It is now read off a column-equilibrated copy.

For the basis ORDER, the scaling is exactly the information wanted. The null-space
step asks the BASIC variables to absorb the infeasibility through
`dn_b = B⁻¹(−ew)`, so they must be the ones that can move — the abundant species,
not a trace ion pinned at its bound. Equilibrating before pivoting threw that away:
on an LC³ equilibrium the basis then held species at 1e-16, the particular solution
asked them for 1e4 mol, and the dual step came back at 1e31. The order is now taken
from the matrix as given, which is how Optima prioritizes its own basis.

A genuinely rank-deficient conservation matrix is reported as such, naming how
many constraints are redundant, instead of surfacing as a factorization error.

### The reduced Hessian was not equilibrated, and the default step was noise

`compute_step!` equilibrates its Schur complement, with a comment explaining why.
`compute_step_nullspace!` — which is the DEFAULT path through `OptimaOptimizer` —
did not equilibrate its reduced Hessian `Zᵀ H Z = Rᵀ diag(h_b) R + diag(h_n)`.

`h` is the barrier-augmented curvature `∇²f + μ/s²`, and in a chemical system the
amounts span ten orders of magnitude, so `h` spans twenty and more: on a cement it
ran from 2.5 on the solvent to 1e27 on a species at its bound. The condition number
of the reduced Hessian went past anything Float64 can carry, and the Cholesky then
*succeeded* while returning a direction that was noise — `‖dn‖∞ = 4.5e17`,
`‖dy‖∞ = 3.1e43`, and `NaN` two iterations later. Scaling by the square root of the
diagonal is exact and costs nothing.

### The starting point was never feasible, and the line search could not recover

The starting point was projected onto `A n = b` by a single minimum-norm
correction and then clamped to the bounds, which puts it straight back off the
affine set — and with most candidate species at their lower bound the clamp
restores a large amount of matter. Positivity is now enforced FIRST and
feasibility after, so nothing undoes it.

That reordering is not enough on its own, because the projection has to be exact.
The filter line search bypasses its filter only when the current point is
feasible; while it is not, acceptance needs either a relative drop of `ls_alpha`
in the constraint violation — unreachable once the fraction-to-boundary limit is
itself below `ls_alpha` — or an Armijo decrease along a direction that is partly
spent restoring feasibility and need not be a descent direction at all. Measured
on an LC³ equilibrium the start carried `‖An − b‖∞ = 6.8e-3`, all forty trial
steps were refused at every barrier level, and the solve reported `MaxIters` on
the point it had started from, having never moved.

Feasibility is now attained rather than approached, by two exact routes before the
old fallback:

  - solve for the BASIC amounts given the others, `B n_b = b − N n_n` — one
    triangular solve on a factorization that already exists, which is how Optima
    does it;
  - failing that (a component total can be negative — the `H⁺` row of a cement is
    −2.1 mol — and no basis of abundant species can produce it), Lawson–Hanson
    **non-negative least squares** on the slacks `v = n − lb`. It terminates
    finitely and its residual is zero whenever the budget is attainable, which it
    is, since the budget came from a real composition.

NNLS returns a solution supported on at most `rank(A)` variables, with the rest at
exactly their bound and zero slack — unusable as a barrier start, since the first
negative step component would give `α = 0`. They are lifted to the slack the
barrier itself would give them, `s = μ/(∇f)ᵢ`, about 1e-6 for the initial
`μ = 1e-4`; the matter that adds is then removed exactly, on the support, by a
minimum-norm correction, so the point is both feasible to machine precision and
strictly interior.

Together with the two items above, on that LC³ equilibrium: the start goes from
`‖An − b‖∞ = 0.86` to `3.4e-10`, the step lengths from 1e-19 to 0.3–0.75, the
feasibility error stays at 3e-15 for the whole solve because the null-space step
preserves it, and the optimality error falls from 199 to 3e-4. Before these
changes it did not move at all.

### Verbose output reports what actually stopped the step

`log_iteration` now prints the fraction-to-boundary limit `α_max` beside the
accepted `α`, and the step norms `‖dn‖∞`, `‖dy‖∞`. The two failures those separate
are indistinguishable from `α` alone — a step the filter refuses looks exactly like
a step the bounds never allowed — and telling them apart is what located every one
of the defects above.

## v0.3.0 — a KKT solver that proves its answer

### Breaking changes

Five new exported names — `DualNewtonProblem`, `DualNewtonOptions`,
`dual_newton_solve`, `kkt_certificate`, `degenerate_components`. Nothing was
removed or renamed and no existing signature changed, but below 1.0 the resolver
treats a minor bump as breaking regardless, so a downstream
`[compat] OptimaSolver = "0.2"` must be widened.

### `dual_newton_solve`: Newton on the KKT system in multiplier space

The interior-point method of `solve!` minimizes `f` by walking the interior, and
on a problem whose bounded variables have zero curvature it does not reach its
tolerance: the fraction-to-boundary rule caps the step at every iteration. That
is intrinsic to a primal barrier method.

The alternative solves the KKT conditions directly. With `u = −Aᵀy`, an interior
variable obeys `hᵢ(x) = uᵢ − gᵢ` — invertible, a mass-action law in a chemical
system — and a bounded variable is positive exactly when `gᵢ = uᵢ`, zero when
`gᵢ ≥ uᵢ`. That second line is a stability criterion, and the active set on it is
finite for a convex problem.

Parameterizing the interior variables by `ln x` makes their positivity automatic,
so the fraction-to-boundary rule has nothing left to act on. The outer system is
`1 + m + |P|` unknowns, some fourteen for a cement partition against forty-seven
variables in the interior-point route.

This is **not** the log reparameterization of `variable_space = Val(:log)`:
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
  every solve from the end state, returning pH 14.2 where honoring the guess
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
