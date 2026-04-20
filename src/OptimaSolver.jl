# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

"""
    OptimaSolver

Julia-native primal-dual interior-point solver for Gibbs-energy minimisation.

Implements the Optima algorithm (Allan Leal, ETH Zürich) in Julia, with:
- full ForwardDiff / AD compatibility (no Float64 casts)
- Schur-complement Newton step exploiting diagonal Hessian structure
- filter line search (Wächter & Biegler 2006)
- implicit-differentiation sensitivity ∂n*/∂(b, μ⁰/RT)
- warm-start between consecutive solves
- drop-in SciML interface (`OptimaOptimizer`) compatible with ChemistryLab.jl

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
export KKTResidual, kkt_residual, hessian_diagonal, gibbs_hessian_diag
export NewtonStep, compute_step!, clamp_step
export LineSearchFilter, line_search
export classify_variables, reduced_step_for_unstable!, stability_measure

end # module OptimaSolver
