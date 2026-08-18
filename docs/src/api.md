```@meta
CurrentModule = OptimaSolver
```

# API Reference

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
