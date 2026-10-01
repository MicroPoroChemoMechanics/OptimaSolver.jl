```@meta
CurrentModule = OptimaSolver
```

# API Reference

```@docs
OptimaSolver
```

## Problem definition

```@docs
OptimaProblem
OptimaOptions
OptimaState
OptimaResult
```

## Canonicalizer

```@docs
Canonicalizer
```

## Solver

`solve` is exported and re-exports `SciMLBase.solve`, so `using OptimaSolver` is
sufficient. `solve!` (the in-place variant) is not exported; use the qualified name
`OptimaSolver.solve!(...)` or `import OptimaSolver: solve!`.

```@docs
OptimaSolver.solve
OptimaSolver.solve!
```

## KKT solver and certificate

Newton on the KKT conditions, and the proof that a point is optimal. Derived in
[Newton on the KKT system in multiplier space](@ref).

```@docs
SolutionPhase
DualNewtonProblem
DualNewtonOptions
dual_newton_solve
dual_newton_tangent
kkt_certificate
degenerate_components
stationarity_capacity
OptimaSolver.DEGENERATE_POTENTIAL
```

The tangent-plane measures the search and the certificate apply to a mixing
phase, absent or present:

```@docs
phase_tangent_trial
phase_tangent_measure
phase_split_trial
phase_split_measure
```

## Linear-programming start

A start, or a proof that the budget cannot be met. Derived in
[The linear program over pure phases](@ref).

```@docs
lp_start
LPStart
simplex_start
```

## Sensitivity

```@docs
SensitivityResult
sensitivity
```

## SciML interface

```@docs
OptimaOptimizer
reset_cache!
```

## Internal components

The following symbols are exported for testing and extension purposes.
They are not needed for typical usage.

### KKT residual and Hessian

```@docs
KKTResidual
kkt_residual
OptimaSolver.row_scales
hessian_diagonal
gibbs_hessian_diag
```

### Convergence and the barrier schedule

The optimality error and the barrier update are derived in
[The optimality error, and why the obvious one cannot work](@ref).

```@docs
OptimaSolver.is_converged
OptimaSolver.should_reduce_barrier
OptimaSolver.reduce_barrier
```

### Newton step

```@docs
NewtonStep
compute_step!
OptimaSolver.compute_step_nullspace!
clamp_step
```

### Line search

```@docs
LineSearchFilter
line_search
```

### Variable stability

```@docs
classify_variables
reduced_step_for_unstable!
stability_measure
```
```
