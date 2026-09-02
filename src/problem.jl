# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── problem.jl ─────────────────────────────────────────────────────────────────
# Core data structures: OptimaProblem, OptimaState, OptimaResult, OptimaOptions

"""
    OptimaProblem{T, F, G}

Gibbs-energy minimization problem in the form:

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

# Element type

`T` is promoted over `A`, `b`, `lb`, `ub` **and the numeric leaves of `p`**. The
last one is what makes forward-mode differentiation with respect to a parameter
work: `ForwardDiff` seeds `p` with `Dual` numbers, and if `T` were inferred from
the constraint data alone it would stay `Float64`, so the solver's gradient
buffer would be a `Vector{Float64}` and the user's `g!` would fail trying to
write a `Dual` into it.
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

"""
    param_eltype(p) -> Type

Numeric element type carried by the parameter object `p`, or `Union{}` when it
carries none. Used to promote an `OptimaProblem`'s element type, so that a
parameter seeded with `ForwardDiff.Dual` numbers propagates into every workspace
the solver allocates.

Descends into tuples and named tuples, which is how parameters are normally
passed (`p = (μ⁰ = ..., T = ...)`). Anything it does not recognize as numeric
contributes `Union{}`, which is neutral for `promote_type`.
"""
param_eltype(::Nothing) = Union{}
param_eltype(x::Number) = _concrete_numeric(typeof(x))
param_eltype(x::AbstractArray) = _concrete_numeric(eltype(x))
function param_eltype(x::Union{Tuple, NamedTuple})
    return mapreduce(param_eltype, promote_type, values(x); init = Union{})
end
param_eltype(::Any) = Union{}

"""
    _concrete_numeric(T) -> Type

`T` if it is a concrete number type, `Union{}` otherwise.

The concreteness test is what keeps this safe, and it is not a detail. A
parameter may perfectly well carry an abstractly typed container — the SciML
path of `ChemistryLab` passes `A::Matrix{Real}` and `b::Vector{Any}` inside `p`,
because its stoichiometric matrices are built that way. `eltype` of those is
`Real` and `Any`, which name no working precision: promoting to them made
`OptimaProblem{Real}`, and the solver died on `eps(Real)`, which has no method.

Only a concrete type can be the arithmetic the solver runs in — and that is
exactly what the case this promotion exists for provides, since a
`ForwardDiff.Dual` seeded into `p` is always concretely typed.
"""
_concrete_numeric(::Type{T}) where {T} = (isconcretetype(T) && T <: Number) ? T : Union{}

function OptimaProblem(
        A::AbstractMatrix,
        b::AbstractVector,
        f::F,
        g!::G;
        lb::AbstractVector = fill(1.0e-16, size(A, 2)),
        ub::AbstractVector = fill(Inf, size(A, 2)),
        p = nothing,
    ) where {F <: Function, G <: Function}
    T = promote_type(eltype(A), eltype(b), eltype(lb), eltype(ub), param_eltype(p))
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

!!! warning "`use_fd_hessian` defaults differently here and on `OptimaOptimizer`"
    This struct defaults it to `false`; the `OptimaOptimizer(; …)` keyword
    constructor defaults it to `true`. Building the same optimizer the two ways
    therefore selects **opposite regimes**, and neither is right everywhere.

    Measured on chemical equilibria, where the amounts span ten orders of
    magnitude between the solvent and the trace ions:

    | | pure water | mixed solid/aqueous |
    |:--|--:|--:|
    | `false` (analytic 1/nᵢ) | `[H⁺]/[OH⁻] = 1.000003` ✓ | worst ×6181 ✗ |
    | `true` (finite difference) | `[H⁺]/[OH⁻] = 3.78` ✗ | worst ×19.8 |

    A Hessian affects the *rate* of Newton's method, not its limit, so a solver
    that merely converged slowly would still reach the right answer. It does
    not — the iteration stalls, identically at 300 and at 200 000 iterations —
    which places the fault in the interior-point loop, most likely in how a
    pure phase (whose objective is linear in its amount) is driven to its
    bound. Until that is resolved, choose this flag deliberately rather than
    relying on either default.
"""
Base.@kwdef struct OptimaOptions
    tol::Float64 = 1.0e-10
    max_iter::Int = 300
    warm_start::Bool = true
    barrier_init::Float64 = 1.0e-4
    barrier_min::Float64 = 1.0e-14
    # κ_μ of Wächter & Biegler (2006), Eq. (7). Ipopt uses 0.2; 0.1 is more
    # aggressive, and with the convergence test taken at μ = 0 it is too
    # aggressive to converge at all — the barrier outruns the inner Newton and
    # the error plateaus just above the tolerance. See `should_reduce_barrier`.
    barrier_decay::Float64 = 0.2
    # How far above μ the barrier subproblem may be left before μ is reduced.
    # Ipopt's κ_ε (Wächter & Biegler 2006, Eq. 7) is 10, but applies to their
    # SCALED error; ours is unscaled, so the default here is 1. See
    # `should_reduce_barrier`.
    barrier_eps_factor::Float64 = 1.0
    # Consecutive inner iterations without a relative improvement of 1e-3 after
    # which the barrier is reduced anyway. `typemax` disables the rule.
    barrier_stall_iters::Int = 8
    ls_alpha::Float64 = 1.0e-4
    ls_beta::Float64 = 0.5
    ls_max_iter::Int = 40
    verbose::Bool = false
    use_fd_hessian::Bool = false
    nullspace_step::Bool = true
end

# ── OptimaState ───────────────────────────────────────────────────────────────

"""
    OptimaState{T}

Mutable solver state — primal variables `n`, dual variables `y` (Lagrange
multipliers for A n = b), and the barrier parameter `μ`.

Warm-starting: pass the converged state from a previous solve as `u0` to
`solve`; the solver will initialize (n, y) from it.
"""
mutable struct OptimaState{T <: Real}
    n::Vector{T}       # primal: mole amounts (ns,)
    y::Vector{T}       # dual: Lagrange multipliers for A n = b (m,)
    μ::T               # log-barrier weight
    iter::Int          # iteration count
    converged::Bool
    error_opt::T       # ‖∇G + Aᵀy - μ/n‖∞  (optimality)
    error_feas::T      # feasibility, each row scaled by its own budget
    error_feas_abs::T  # ‖An - b‖∞, unscaled, in the units of `b`
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
        T(Inf),
    )
end

function OptimaState(n0::AbstractVector{T}, y0::AbstractVector{T}, μ0::T) where {T}
    return OptimaState{T}(
        copy(n0), copy(y0), μ0, 0, false, T(Inf), T(Inf), T(Inf),
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
- `error_feas`: final feasibility residual, each row scaled by its own budget
  (see `row_scales`) — this is the quantity `tol` is compared against
- `error_feas_abs`: the same residual unscaled, `‖An − b‖∞`, for reporting
"""
struct OptimaResult{T <: Real}
    n::Vector{T}
    y::Vector{T}
    iterations::Int
    converged::Bool
    error_opt::T
    error_feas::T
    error_feas_abs::T
end

function OptimaResult(state::OptimaState{T}) where {T}
    return OptimaResult{T}(
        copy(state.n),
        copy(state.y),
        state.iter,
        state.converged,
        state.error_opt,
        state.error_feas,
        state.error_feas_abs,
    )
end
