# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

"""
    OptimaSolver

Gibbs-energy minimization for equilibrium chemistry, by two solvers.

The interior-point method of [`solve`](@ref), a Julia port of the Optima library
(Allan Leal, ETH Zürich), with:
- Schur-complement Newton step exploiting diagonal Hessian structure
- filter line search (Wächter & Biegler 2006)
- implicit-differentiation sensitivity ∂n*/∂(b, μ⁰/RT)
- warm-start between consecutive solves

Newton's method on the KKT conditions in the space of the conservation
multipliers, [`dual_newton_solve`](@ref), original to this package, with an
active set over the pure phases, the KKT certificate
[`kkt_certificate`](@ref), Michelsen's phase-stability tests, a start from the
linear program over pure phases ([`lp_start`](@ref)) and the derivatives of the
answer ([`dual_newton_tangent`](@ref)).

Both are written in generic arithmetic (ForwardDiff dual numbers pass through),
and `OptimaOptimizer` is a drop-in SciML interface compatible with ChemistryLab.jl.

# Main entry points
- [`OptimaProblem`](@ref)       — problem definition
- [`OptimaOptions`](@ref)       — solver hyperparameters
- [`solve`](@ref)               — main solve function
- [`sensitivity`](@ref)         — post-convergence sensitivity matrices
- [`OptimaOptimizer`](@ref)     — SciML drop-in optimizer
"""
module OptimaSolver

using LinearAlgebra
import ForwardDiff
import SciMLBase
import SciMLBase: solve

# ── Source files (dependency order) ──────────────────────────────────────────
include("problem.jl")           # OptimaProblem, OptimaState, OptimaResult, OptimaOptions
include("canonicalizer.jl")     # Canonicalizer — A → [B N], LU cache, Schur complement
include("dual_newton.jl")
include("dual_newton_ad.jl")
include("lp.jl")                # lp_start, LPStart: the linear program over pure phases
include("residual.jl")          # KKTResidual, kkt_residual, hessian_diagonal
include("newton_step.jl")       # NewtonStep, compute_step!, clamp_step
include("stability.jl")         # classify_variables, reduced_step_for_unstable!
include("line_search.jl")       # LineSearchFilter, line_search
include("convergence.jl")       # is_converged, reduce_barrier, log_iteration
include("sensitivity.jl")       # SensitivityResult, sensitivity
include("solver.jl")            # solve!, solve
include("sciml_interface.jl")   # OptimaOptimizer, SciMLBase.solve

# ── Exports ───────────────────────────────────────────────────────────────────

# Problem definition
export OptimaProblem, OptimaOptions, OptimaState, OptimaResult

# Canonicalizer (exposed for reuse across solves with fixed A)
export Canonicalizer

# Solver  (solve is exported and re-exports SciMLBase.solve;
# solve! is internal — accessible as OptimaSolver.solve!)

# Sensitivity
export SensitivityResult, sensitivity

# SciML drop-in
export OptimaOptimizer, reset_cache!, solve

# Internal components (exported for testing and extension)
export SolutionPhase, DualNewtonProblem, DualNewtonOptions, dual_newton_solve,
    dual_newton_tangent,
    kkt_certificate, degenerate_components, stationarity_capacity,
    phase_tangent_measure,
    phase_tangent_trial,
    phase_split_measure,
    phase_split_trial,
    simplex_start,
    lp_start, LPStart,
    KKTResidual, kkt_residual, hessian_diagonal, gibbs_hessian_diag
export row_scales
export NewtonStep, compute_step!, compute_step_nullspace!, clamp_step
export LineSearchFilter, line_search
export classify_variables, reduced_step_for_unstable!, stability_measure

end # module OptimaSolver
