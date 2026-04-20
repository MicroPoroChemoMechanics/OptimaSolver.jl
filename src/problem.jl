# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── problem.jl ─────────────────────────────────────────────────────────────────
# Core data structures: OptimaProblem, OptimaState, OptimaResult, OptimaOptions

"""
    OptimaProblem{T, F, G}

Gibbs-energy minimisation problem in the form:

    minimize    f(n, p)            (e.g. G(n) = nᵀ μ(n,p))
    subject to  A n = b            (mass conservation, m × ns)
                n ≥ ε              (positivity)

# Fields
- `A`:  conservation matrix (m × ns), typically integer-valued
- `b`:  RHS vector (m,)
- `f`:  objective function `(n, p) -> scalar`
- `g!`: in-place gradient `(grad, n, p) -> nothing`  (∂f/∂n)
- `ns`: number of species
- `m`:  number of conservation equations
- `lb`: lower bounds on n (default: fill(ε, ns))
- `ub`: upper bounds on n (default: fill(Inf, ns))
- `p`:  parameter tuple passed through to f and g!
"""
struct OptimaProblem{T <: Real, F <: Function, G <: Function}
    A::Matrix{T}
    b::Vector{T}
    f::F
    g!::G
    ns::Int
    m::Int
    lb::Vector{T}
    ub::Vector{T}
    p::Any
end

function OptimaProblem(
        A::AbstractMatrix,
        b::AbstractVector,
        f::F,
        g!::G;
        lb::AbstractVector = fill(1.0e-16, size(A, 2)),
        ub::AbstractVector = fill(Inf, size(A, 2)),
        p = nothing,
    ) where {F <: Function, G <: Function}
    T = promote_type(eltype(A), eltype(b), eltype(lb), eltype(ub))
    ns = size(A, 2)
    m = size(A, 1)
    @assert length(b) == m "b length $(length(b)) must match A rows $m"
    @assert length(lb) == ns && length(ub) == ns "bounds must have length $ns"
    return OptimaProblem{T, F, G}(
        convert(Matrix{T}, A),
        convert(Vector{T}, b),
        f, g!,
        ns, m,
        convert(Vector{T}, lb),
        convert(Vector{T}, ub),
        p,
    )
end

# ── OptimaOptions ─────────────────────────────────────────────────────────────

"""
    OptimaOptions

Solver hyperparameters.

# Fields
- `tol`:           KKT residual tolerance (default 1e-10)
- `max_iter`:      maximum Newton iterations (default 300)
- `warm_start`:    reuse previous (n, y) as initial guess (default true)
- `barrier_init`:  initial log-barrier weight μ₀ (default 1e-4)
- `barrier_min`:   minimum barrier weight (default 1e-14)
- `barrier_decay`: barrier reduction factor per outer iteration (default 0.1)
- `ls_alpha`:      Armijo sufficient-decrease parameter (default 1e-4)
- `ls_beta`:       backtracking contraction factor (default 0.5)
- `ls_max_iter`:   maximum backtracking steps (default 40)
- `verbose`:          print iteration log (default false)
- `use_fd_hessian`:   compute Hessian diagonal via finite differences of ∇f
                      instead of the ideal-solution approximation 1/nᵢ
                      (default false). Enable for problems with pure solid or
                      gas species where the true ∂²f/∂nᵢ² = 0, otherwise the
                      approximation 1/nᵢ causes extremely slow convergence.
"""
Base.@kwdef struct OptimaOptions
    tol::Float64 = 1.0e-10
    max_iter::Int = 300
    warm_start::Bool = true
    barrier_init::Float64 = 1.0e-4
    barrier_min::Float64 = 1.0e-14
    barrier_decay::Float64 = 0.1
    ls_alpha::Float64 = 1.0e-4
    ls_beta::Float64 = 0.5
    ls_max_iter::Int = 40
    verbose::Bool = false
    use_fd_hessian::Bool = false
end

# ── OptimaState ───────────────────────────────────────────────────────────────

"""
    OptimaState{T}

Mutable solver state — primal variables `n`, dual variables `y` (Lagrange
multipliers for A n = b), and the barrier parameter `μ`.

Warm-starting: pass the converged state from a previous solve as `u0` to
`solve`; the solver will initialise (n, y) from it.
"""
mutable struct OptimaState{T <: Real}
    n::Vector{T}       # primal: mole amounts (ns,)
    y::Vector{T}       # dual: Lagrange multipliers for A n = b (m,)
    μ::T               # log-barrier weight
    iter::Int          # iteration count
    converged::Bool
    error_opt::T       # ‖∇G + Aᵀy - μ/n‖∞  (optimality)
    error_feas::T      # ‖An - b‖∞           (feasibility)
end

function OptimaState(ns::Int, m::Int, T::Type = Float64)
    return OptimaState{T}(
        fill(one(T), ns),
        zeros(T, m),
        T(1.0e-4),
        0,
        false,
        T(Inf),
        T(Inf),
    )
end

function OptimaState(n0::AbstractVector{T}, y0::AbstractVector{T}, μ0::T) where {T}
    return OptimaState{T}(
        copy(n0), copy(y0), μ0, 0, false, T(Inf), T(Inf),
    )
end

# ── OptimaResult ──────────────────────────────────────────────────────────────

"""
    OptimaResult{T}

Immutable solver output.

# Fields
- `n`:          equilibrium mole amounts
- `y`:          Lagrange multipliers at convergence
- `iterations`: total Newton iterations
- `converged`:  convergence flag
- `error_opt`:  final KKT optimality residual
- `error_feas`: final feasibility residual
"""
struct OptimaResult{T <: Real}
    n::Vector{T}
    y::Vector{T}
    iterations::Int
    converged::Bool
    error_opt::T
    error_feas::T
end

function OptimaResult(state::OptimaState{T}) where {T}
    return OptimaResult{T}(
        copy(state.n),
        copy(state.y),
        state.iter,
        state.converged,
        state.error_opt,
        state.error_feas,
    )
end
