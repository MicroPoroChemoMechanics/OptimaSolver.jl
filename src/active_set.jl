# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── active_set.jl ──────────────────────────────────────────────────────────────
#
# The active-set iteration of Optima (reaktoro/optima, `MasterSolver`): an
# active set on the lower bounds, not a barrier schedule, with the design choices
# `solve_active_set` lists.

"""
    solve_active_set(prob, opts; u0 = nothing, y0 = nothing) -> OptimaResult

Minimize `f` subject to `A n = b` and `lb ≤ n ≤ ub` by the active-set Newton
iteration of Optima [Leal2024optima](@cite), the solver Reaktoro computes its
equilibria with, from the amounts `u0` and the multipliers `y0`.
`opts.algorithm == :active_set` selects it in [`solve`](@ref); the interior point
stays available as `:interior_point`.

# One iteration

At the amounts `n` and the multipliers `y`, with `g = ∇f(n) − τ ./ n` and
`s = g + Aᵀy` the reduced gradient:

1. **Basis.** `m` basic variables: a basis of `A` by pivoted QR, then each
   basic variable exchanged for the heaviest non-basic one that can replace it
   (weight `nᵢ`, the lowest for a variable on its bound): the most abundant
   species carry the balance, on a basis `A` supports.
2. **Stability.** A non-basic variable on its lower bound whose `sᵢ > 0` would
   decrease if it could: it is *unstable*, held on its bound (`dxᵢ = 0`) and left
   out of the error. A basic variable is never held, so that every row keeps a
   variable that can restore it.
3. **Newton step** on the KKT system of the stable variables, the residual of the
   balance `A n − b` in its right-hand side, solved in the null space of the
   stable columns of `A` (Optima's default): a full step restores the balance.
4. **Back to the bounds.** The step is shortened to the first variable it would
   take below its bound among those that are not negligible, applied to `n` and
   `y`, and `n` is clamped into `[lb, ub]`. It is accepted: there is no line
   search.
5. **Convergence** when the larger of the optimality error `‖s‖∞` (stable
   variables, basic variables on a bound excepted) and the balance error is below
   `opts.tol`.

# Design choices

- **Every error is relative.** The barrier is `τ = opts.barrier_rel · max|b|`
  (`1e-20` by default; at `1e-16` it moved a trace such as `H⁺` at pH 13 by 0.1 RT
  in its potential), and the balance is judged row by row, relative to what the
  row holds, a row without a budget (an element absent from the system) relative
  to the scale of the whole budget. The answer and the iteration count do not
  depend on how much matter the system holds.
- **A saddle point is not an answer.** On a non-convex Gibbs energy (a solid
  solution with a miscibility gap) Newton's method with the exact Hessian converges
  as readily to a saddle point as to a minimum. The reduced Hessian is tested at
  convergence; where it has a negative curvature, the solve is made again from the
  same start with Ipopt's inertia correction [WachterBiegler2006](@cite): the
  Hessian shifted by `δ diag(1/n)`, `δ` raised until the reduced Hessian is
  positive definite.
- **A negligible amount does not shorten the step.** A variable below
  `opts.negligible_rel · max|b|` (`1e-12` by default) is set on its bound instead,
  the balance it then misses restored by the next step, whose right-hand side
  holds it: a start holding a hundred traces sets them on their bounds together.
- **The Hessian may come from the caller.** A function `p.hess(H, n)` among the
  parameters fills the Hessian of `f` in place of its forward-mode Jacobian of the
  gradient, which costs as many gradient evaluations as there are variables,
  divided by the chunk. ChemistryLab hands over the exact diagonal and the ideal
  coupling within each phase. The saddle test at convergence reads the exact one.
- **A component the iterate holds none of is priced by the dual ratio test.** In
  a row of the canonical form whose variables are all on their bounds, no amount
  fixes the multiplier, and the basic variable that would fix it by its own
  stationarity can make another carrier read as supersaturated: the active set
  cycled on such rows, the reduced sulfur and hydrogen of a cement paste. The
  multiplier is moved, and the basic variable chosen, so that every carrier keeps
  a non-negative reduced gradient (`_degenerate_pivots!`).
- **A cycle ends.** On a Hessian that is not exact, the active set can cycle. A
  descent only lowers the error, and a run that leaves a saddle point or a poor
  first step raises it before lowering it; a cycle lowers it and raises it again.
  After ten such turns without halving its best error, the run ends on its best
  iterate.

A solve that does not converge is made once more, its steps
clamped into the bounds rather than shortened. The best iterate is returned.

Iterative and branching on comparisons of its iterates: real numbers, dual numbers
included, not symbolic ones.
"""
function solve_active_set(prob::OptimaProblem{T}, opts::OptimaOptions; u0 = nothing, y0 = nothing) where {T}
    u0, y0 = _as_start(u0, y0)
    plain = _active_set_run(prob, opts, u0, y0; clamp = false, inertia = false)
    best = plain
    if !plain.converged
        retry = _active_set_run(prob, opts, u0, y0; clamp = true, inertia = false)
        _active_set_error(retry) < _active_set_error(best) && (best = retry)
    end
    if best.converged && best.saddle
        escaped = _active_set_run(prob, opts, u0, y0; clamp = false, inertia = true)
        escaped.converged && !escaped.saddle && (best = escaped)
    end
    opts.verbose && println(
        "  [OptimaSolver] ", best.converged ? "CONVERGED" : "MAX_ITER", " after ", best.iterations,
        " iterations | err = ", _fmt_sci(_active_set_error(best)), best.converged && best.saddle ? " | saddle point" : "",
    )
    return OptimaResult{T}(best.n, best.y, best.iterations, best.converged, best.err_x, best.err_w, best.err_w_abs)
end

_active_set_error(r) = max(r.err_x, r.err_w, r.err_v)

"""
    solve_active_set(prob::OptimaParametricProblem, opts; u0 = nothing, q0 = nothing, y0 = nothing)
        -> OptimaParametricResult

The same iteration on Optima's master problem with unknown parameters `q`
started from the amounts `u0`, the parameters `q0` (zero, clamped
into their bounds, by default) and the multipliers `y0`.

The basic variables are chosen among the amounts, and the parameters enter the
step through `S = B⁻¹ Ap`: with `Z = [−R; I]` the null space of the stable
columns and `Zₚ = [−S; 0]`, the step `(dz, dq)` of the free amounts and of the
parameters solves

    [ ZᵀHZ        Zᵀ(HZₚ + Hₙq) ] [dz]   [ −Zᵀ(g + H dx₀)   ]
    [ VₙZ         VₙZₚ + V_q    ] [dq] = [ −v − Vₙ dx₀      ]

where `dx₀` restores the balance on the basic variables, `Hₙq = ∂g/∂q` and
`(Vₙ, V_q)` the Jacobian of `v`. The basic amounts follow,
`dx_B = dx₀ − R dz − S dq`, and the multipliers from the basic rows. Convergence
asks the largest residual of `v` below `opts.tol` as well. With no parameter the
step is that of [`OptimaProblem`](@ref), operation for operation.

The bounds on `q` are applied by clamping after the step.
"""
function solve_active_set(prob::OptimaParametricProblem{T}, opts::OptimaOptions; u0 = nothing, q0 = nothing, y0 = nothing) where {T}
    plain = _active_set_run(prob, opts, u0, y0; clamp = false, inertia = false, q0 = q0)
    best = plain
    if !plain.converged
        retry = _active_set_run(prob, opts, u0, y0; clamp = true, inertia = false, q0 = q0)
        _active_set_error(retry) < _active_set_error(best) && (best = retry)
    end
    opts.verbose && println(
        "  [OptimaSolver] ", best.converged ? "CONVERGED" : "MAX_ITER", " after ", best.iterations,
        " iterations | err = ", _fmt_sci(_active_set_error(best)),
    )
    return OptimaParametricResult{T}(
        best.n, best.q, best.y, best.iterations, best.converged, best.err_x, best.err_w, best.err_w_abs, best.err_v,
    )
end

# The problems the iteration runs on: without parameters, and with them.
const _ActiveSetProblem{T} = Union{OptimaProblem{T}, OptimaParametricProblem{T}}

# The parameters at the start, clamped into their bounds; none without them.
_start_params(::OptimaProblem{T}, q0) where {T} = T[]
_start_params(prob::OptimaParametricProblem{T}, ::Nothing) where {T} = clamp.(zeros(T, prob.nq), prob.qlb, prob.qub)
function _start_params(prob::OptimaParametricProblem{T}, q0::AbstractVector) where {T}
    length(q0) == prob.nq || throw(DimensionMismatch("`q0` holds $(length(q0)) parameters for $(prob.nq)."))
    return clamp.(convert(Vector{T}, q0), prob.qlb, prob.qub)
end

# `∂f/∂n` at `(n, q)`.
_gradient!(g, prob::OptimaProblem, n, q) = prob.g!(g, n, prob.p)
_gradient!(g, prob::OptimaParametricProblem, n, q) = prob.g!(g, n, q, prob.p)

# The residual of the linear constraints.
_balance_residual(prob::OptimaProblem, n, q) = prob.A * n .- prob.b
_balance_residual(prob::OptimaParametricProblem, n, q) = prob.A * n .+ prob.Ap * q .- prob.b

# The derivatives the parameters add to the step at `(n, q)`: `∂g/∂q`, the
# residual `v` and its Jacobian, by forward mode; `nothing` without parameters.
_parameter_blocks(::OptimaProblem, n, q) = nothing
function _parameter_blocks(prob::OptimaParametricProblem{T}, n, q) where {T}
    ns, nq = prob.ns, prob.nq
    Hnq = ForwardDiff.jacobian(qq -> _gradient!(similar(qq, ns), prob, n, qq), q)
    vz = z -> (vv = similar(z, nq); prob.v!(vv, view(z, 1:ns), view(z, (ns + 1):(ns + nq)), prob.p); vv)
    z = vcat(n, q)
    v = vz(z)
    V = ForwardDiff.jacobian(vz, z)
    return (; Hnq, v, Vn = V[:, 1:ns], Vq = V[:, (ns + 1):end])
end

# The largest residual of `v`, zero without parameters.
_parameter_error(::Nothing, ::Type{T}) where {T} = zero(T)
_parameter_error(pb, ::Type{T}) where {T} = maximum(abs, pb.v; init = zero(T))

# The turns of its error back up after a fall that a run of the active set may
# take without halving its best error, beyond which it has stalled.
const ACTIVE_SET_STALL_WINDOW = 10

# The start, from a previous answer or its state as `solve` takes them, or from
# the amounts and multipliers given.
_as_start(u0::Union{OptimaResult, OptimaState}, y0) = (u0.n, u0.y)
_as_start(u0, y0) = (u0, y0)

_start_amounts(prob::_ActiveSetProblem, ::Nothing) = _default_initial_n(prob)
_start_amounts(prob::_ActiveSetProblem{T}, u0::AbstractVector) where {T} = clamp.(convert(Vector{T}, u0), prob.lb, prob.ub)

# Multipliers fitted to the stationarity of the most abundant species, or those
# given.
function _start_multipliers(prob::_ActiveSetProblem, g, n, ::Nothing)
    big = partialsortperm(n, 1:min(prob.ns, max(3prob.m, prob.m + 4)); rev = true)
    return LinearAlgebra.qr(transpose(prob.A[:, big]), LinearAlgebra.ColumnNorm()) \ (-g[big])
end
function _start_multipliers(prob::_ActiveSetProblem{T}, g, n, y0::AbstractVector) where {T}
    length(y0) == prob.m || throw(
        DimensionMismatch("`y0` holds $(length(y0)) multipliers for $(prob.m) conservation rows."),
    )
    return convert(Vector{T}, y0)
end

# The pivot below which an entry of the canonical form `S = B⁻¹N` is taken for
# zero: `S` holds ratios of stoichiometric coefficients, so it is dimensionless.
const BASIS_PIVOT_TOL = 1.0e-8

# The `m` basic variables of `A`: a basis of the matrix alone, by pivoted QR,
# then each basic variable exchanged for the heaviest non-basic one that can
# replace it, a non-basic `k` replacing the `i`-th basic variable when
# `|S[i, k]| > BASIS_PIVOT_TOL` in the canonical form `S = B⁻¹N`. The most
# abundant species then carry the balance, and the basis stays one that `A`
# itself supports: weighting the columns before the factorization instead
# lets a column whose weight is near zero (an absent species, the only one in
# its row) lose to a combination of others that is singular in exact
# arithmetic and survives the factorization by rounding.
_weighted_basis(A::AbstractMatrix, w::AbstractVector) = sort(first(_canonical_basis(A, w)))

# The basis of `_weighted_basis`, with the non-basic variables and the canonical
# form `S = B⁻¹N`, row `r` that of the basic variable `jb[r]`, column `c` that of
# the non-basic one `jn[c]`.
function _canonical_basis(A::AbstractMatrix, w::AbstractVector)
    m, ns = size(A)
    jb = LinearAlgebra.qr(A, LinearAlgebra.ColumnNorm()).p[1:m]
    jn = setdiff(1:ns, jb)
    S = isempty(jn) ? zeros(eltype(A), m, 0) : LinearAlgebra.lu(A[:, jb]) \ A[:, jn]
    # One pass over the basic variables, each exchange updating the canonical
    # form the next one reads: a recurrence.
    for i in 1:m
        cands = [c for c in eachindex(jn) if abs(S[i, c]) > BASIS_PIVOT_TOL]
        isempty(cands) && continue
        k = argmax(c -> w[jn[c]], cands)
        w[jb[i]] >= w[jn[k]] && continue
        _exchange!(S, jb, jn, i, k)
    end
    return jb, jn, S
end

# The Gauss–Jordan exchange of the `i`-th basic variable and the `k`-th non-basic
# one, in place on the canonical form and the two index lists.
function _exchange!(S, jb, jn, i, k)
    piv = S[i, k]
    col = S[:, k]
    col[i] = zero(piv)
    S[i, :] ./= piv
    S .-= col .* transpose(S[i, :])
    S[:, k] .= -col ./ piv
    S[i, k] = inv(piv)
    jb[i], jn[k] = jn[k], jb[i]
    return S
end

"""
    _degenerate_pivots!(y, s, jb, jn, S, at_lb, A)

The rows of the canonical form whose variables are all on their bounds: a
component the iterate holds none of, whose multiplier no variable off its bound
fixes. The basic variable of such a row, chosen by the weights among variables
that all weigh the same, would fix it by its own stationarity, and a carrier of
the row can then read as supersaturated (`sᵢ < 0` on its bound) while the
component is absent: the active set cycled on such rows, the reduced sulfur and
hydrogen of a cement paste. The multiplier of the row is moved instead, and its
basic variable chosen, by the ratio test of the dual simplex method: along the
row `c` (`1` for the basic variable, `S[r, j]` for the others), `s ← s − θ c` with
the largest `θ ≤ 0` that keeps every carrier at `s ≥ 0`, and the carrier that
binds becomes basic. Applied to `y` and `s` in place; `jb`, `jn` and `S` follow
the exchange. A row where no such `θ` exists is left as it is.
"""
function _degenerate_pivots!(y, s, jb, jn, S, at_lb, A)
    for r in eachindex(jb)
        at_lb[jb[r]] || continue
        cs = [c for c in eachindex(jn) if abs(S[r, c]) > BASIS_PIVOT_TOL]
        all(at_lb[jn[c]] for c in cs) || continue
        any(s[jn[c]] < 0 for c in cs) || continue
        θs = Dict(c => s[jn[c]] / S[r, c] for c in cs)
        pos = [c for c in cs if S[r, c] > 0]
        isempty(pos) && continue
        k = argmin(c -> θs[c], pos)
        θ = min(zero(eltype(s)), θs[k])
        all(θ >= θs[c] for c in cs if S[r, c] < 0) || continue
        # y ← y − θ B⁻ᵀ eᵣ moves every reduced gradient by −θ along the row.
        e = zeros(eltype(y), length(jb))
        e[r] = one(eltype(y))
        y .-= θ .* (transpose(LinearAlgebra.lu(A[:, jb])) \ e)
        s[jb[r]] -= θ
        for c in eachindex(jn)
            s[jn[c]] -= θ * S[r, c]
        end
        _exchange!(S, jb, jn, r, k)
    end
    return y, s
end

# The balance error, row by row: relative to what the row holds for a row with a
# budget, to the scale of the whole budget for a row without one (an element
# absent from the system), whose residual is the amounts held on their bound;
# as the dual Newton judges it (`_judged_scales`).
function _balance_error(ew, A, b, n)
    bscale = _system_scale(b)
    held = row_scales(A, b, n)
    return maximum((abs(ew[k]) / (_has_budget(b[k], bscale) ? held[k] : bscale) for k in eachindex(ew)); init = zero(eltype(ew)))
end

# The Newton step in the null space of the stable columns of `A`: the basic
# variables `jb` take the particular solution of `A dx = −ew` and follow the
# free non-basic ones `jn` through `R = B⁻¹ A_N`, whose step minimizes the
# quadratic model on the reduced Hessian `Hr = Zᵀ H Z`, `Z = [−R; I]`. Returns
# `nothing` on a singular basis.
function _reduced_system(H, g, ew, A, jb, jn)
    length(jb) == size(A, 1) || return nothing
    BF = LinearAlgebra.lu(A[:, jb]; check = false)
    LinearAlgebra.issuccess(BF) || return nothing
    R = BF \ A[:, jn]
    dxp = BF \ (-ew)
    Hbb, Hbn = H[jb, jb], H[jb, jn]
    Hnb, Hnn = H[jn, jb], H[jn, jn]
    Hr = Hnn .- Hnb * R .- transpose(R) * Hbn .+ transpose(R) * Hbb * R
    Hr .= (Hr .+ transpose(Hr)) ./ 2
    # −Zᵀ(g + H dxₚ) over the stable variables
    rb = g[jb] .+ Hbb * dxp
    rn = g[jn] .+ Hnb * dxp
    rhs = transpose(R) * rb .- rn
    return (; BF, R, dxp, Hr, rhs)
end

# The reduction of `diag(1/n)`, the curvature of an ideal solution, on the same
# null space: the metric of the inertia correction.
function _reduced_metric(R, n, jb, jn)
    db = one(eltype(n)) ./ n[jb]
    return LinearAlgebra.Diagonal(one(eltype(n)) ./ n[jn]) .+ transpose(R) * (db .* R)
end

# The Hessian a caller hands over through the parameters (`p.hess`, a function
# `(H, n) -> H` of the amounts), as `p.hdiag` for the interior point, or `nothing`.
_hessian_callback(p::NamedTuple) = haskey(p, :hess) ? p.hess : nothing
_hessian_callback(p) = nothing
_hessian_callback(prob::OptimaProblem, q) = _hessian_callback(prob.p)
function _hessian_callback(prob::OptimaParametricProblem, q)
    hess = _hessian_callback(prob.p)
    return hess === nothing ? nothing : (H, n) -> hess(H, n, q)
end

# The Hessian of `f − τ Σ ln nᵢ` at `n`: the caller's plus the barrier's, or the
# forward-mode Jacobian of the gradient.
function _hessian!(H, hess, grad!, gbuf, n, cfg, τ)
    hess(H, n)
    H[LinearAlgebra.diagind(H)] .+= τ ./ n .^ 2
    return H
end
_hessian!(H, ::Nothing, grad!, gbuf, n, cfg, τ) = ForwardDiff.jacobian!(H, grad!, gbuf, n, cfg)

# The step of the free amounts and of the parameters: the reduced system alone
# without parameters, bordered by their rows as `solve_active_set` writes it.
function _free_step(sys, Hr, H, ::Nothing, prob, jb, jn)
    T = eltype(Hr)
    dz = isempty(jn) ? T[] : LinearAlgebra.lu(Hr; check = false) \ sys.rhs
    return dz, T[]
end
function _free_step(sys, Hr, H, pb::NamedTuple, prob::OptimaParametricProblem, jb, jn)
    S = sys.BF \ prob.Ap
    # H Zₚ + Hₙq, and the Jacobian of v along the null space
    HZp = pb.Hnq .- H[:, jb] * S
    Hrq = HZp[jn, :] .- transpose(sys.R) * HZp[jb, :]
    Vz = pb.Vn[:, jn] .- pb.Vn[:, jb] * sys.R
    Vqq = pb.Vq .- pb.Vn[:, jb] * S
    M = [Hr Hrq; Vz Vqq]
    rhs = vcat(sys.rhs, .-pb.v .- pb.Vn[:, jb] * sys.dxp)
    sol = LinearAlgebra.lu(M; check = false) \ rhs
    nn = length(jn)
    return sol[1:nn], sol[(nn + 1):end]
end

# The basic amounts' step, `dx₀ − R dz − S dq`.
_basic_step(sys, dz, dq, ::OptimaProblem) = sys.dxp .- sys.R * dz
_basic_step(sys, dz, dq, prob::OptimaParametricProblem) = sys.dxp .- sys.R * dz .- (sys.BF \ prob.Ap) * dq

# `(g + H dx + Hₙq dq)_B`, the right-hand side of the multipliers.
_basic_gradient(g, H, dx, ::Nothing, dq, jb) = g[jb] .+ (H * dx)[jb]
_basic_gradient(g, H, dx, pb, dq, jb) = g[jb] .+ (H * dx)[jb] .+ (pb.Hnq * dq)[jb]

# The parameters moved by the same fraction of the step, clamped into their bounds.
_advance_params!(q, β, dq, ::OptimaProblem) = q
_advance_params!(q, β, dq, prob::OptimaParametricProblem) = (q .= Base.clamp.(q .+ β .* dq, prob.qlb, prob.qub))

_posdef(M) = LinearAlgebra.issuccess(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(M); check = false))

# One run of the iteration from `(u0, y0)`. `clamp`: the step is clamped into the
# bounds instead of shortened to the first one crossed (the second try).
# `inertia`: the reduced Hessian is shifted to positive definiteness at every
# step.
function _active_set_run(prob::_ActiveSetProblem{T}, opts::OptimaOptions, u0, y0; clamp::Bool, inertia::Bool, q0 = nothing) where {T}
    ns, m = prob.ns, prob.m
    A, b, lb, ub = prob.A, prob.b, prob.lb, prob.ub
    n = _start_amounts(prob, u0)
    # The parameters, updated in place so that the closures below see them
    # without capturing a binding that is reassigned.
    q = _start_params(prob, q0)
    τ = T(opts.barrier_rel) * T(_system_scale(b))
    θ = T(opts.negligible_rel) * T(_system_scale(b))
    grad! = let prob = prob, τ = τ, q = q
        (gg, x) -> begin
            _gradient!(gg, prob, x, q)
            gg .-= τ ./ x
            return gg
        end
    end
    hess = _hessian_callback(prob, q)
    g = zeros(T, ns)
    gbuf = zeros(T, ns)
    H = zeros(T, ns, ns)
    cfg = ForwardDiff.JacobianConfig(grad!, gbuf, n, ForwardDiff.Chunk(n), ForwardDiff.Tag(grad!, eltype(n)))
    grad!(g, n)
    y = _start_multipliers(prob, g, n, y0)

    best = (; n = copy(n), q = copy(q), y = copy(y), err_x = T(Inf), err_w = T(Inf), err_w_abs = T(Inf), err_v = T(Inf))
    converged = false
    saddle = false
    iterations = 0
    progress_ref, turns, err_prev, falling = T(Inf), 0, T(Inf), false
    for it in 1:(opts.max_iter)
        iterations = it
        grad!(g, n)
        s = g .+ transpose(A) * y
        at_lb = n .<= lb
        # on the values: a basis is a choice, not a function to differentiate;
        # a variable on its bound has the lowest priority
        jb, jn_c, S_c = _canonical_basis(_primal_value.(A), [n[i] > lb[i] ? abs(_primal_value(n[i])) : -1.0 for i in eachindex(n)])
        _degenerate_pivots!(y, s, jb, jn_c, S_c, at_lb, A)
        sort!(jb)
        basic = falses(ns)
        basic[jb] .= true
        unstable = .!basic .& at_lb .& (s .> 0)
        ew = _balance_residual(prob, n, q)
        pb = _parameter_blocks(prob, n, q)
        err_x = maximum(i -> (unstable[i] || (basic[i] && at_lb[i])) ? zero(T) : abs(s[i]), 1:ns)
        err_w = _balance_error(ew, A, b, n)
        err_v = _parameter_error(pb, T)
        if max(err_x, err_w, err_v) < _active_set_error(best)
            best = (; n = copy(n), q = copy(q), y = copy(y), err_x, err_w, err_w_abs = maximum(abs, ew), err_v)
        end
        # A search that cycles between active sets lowers its error, then
        # raises it again, over and over, without improving its best iterate.
        # A descent only lowers it, and one that leaves a saddle point or a
        # poor first step raises it before lowering it: neither turns back up
        # after a fall. `STALL_WINDOW` such turns without halving the best
        # error end the run on its best iterate.
        err = max(err_x, err_w, err_v)
        if _active_set_error(best) < T(0.5) * progress_ref
            progress_ref, turns = _active_set_error(best), 0
        elseif falling && err >= err_prev
            turns += 1
            turns >= ACTIVE_SET_STALL_WINDOW && break
        end
        falling, err_prev = err < err_prev, err
        opts.verbose && println(
            "  iter ", lpad(it, 4), " | err_opt = ", _fmt_sci(err_x), " | err_feas = ", _fmt_sci(err_w),
            " | unstable = ", count(unstable),
        )
        jn = [i for i in 1:ns if !basic[i] && !unstable[i]]
        if err <= opts.tol
            converged = true
            # The Hessian of the last step is that of the answer to the
            # tolerance; a negative curvature beyond 1e-8 of that of an ideal
            # solution is a saddle, not rounding.
            if it > 1 && !isempty(jn)
                # on the exact Hessian, once
                ForwardDiff.jacobian!(H, grad!, gbuf, n, cfg)
                sys = _reduced_system(H, g, ew, A, jb, jn)
                saddle = sys !== nothing && !_posdef(sys.Hr .+ T(1.0e-8) .* _reduced_metric(sys.R, n, jb, jn))
            end
            break
        end
        _hessian!(H, hess, grad!, gbuf, n, cfg, τ)
        sys = _reduced_system(H, g, ew, A, jb, jn)
        sys === nothing && break
        Hr = sys.Hr
        if inertia && !isempty(jn)
            D = _reduced_metric(sys.R, n, jb, jn)
            δ = zero(T)
            while !_posdef(Hr .+ δ .* D)
                δ = iszero(δ) ? T(1.0e-12) : 10δ
                δ > T(1.0e6) && break
            end
            Hr = Hr .+ δ .* D
            for i in jb
                H[i, i] += δ / n[i]
            end
        end
        dz, dq = _free_step(sys, Hr, H, pb, prob, jb, jn)
        all(isfinite, dz) && all(isfinite, dq) || break
        dx = zeros(T, ns)
        dx[jn] .= dz
        dx[jb] .= _basic_step(sys, dz, dq, prob)
        # Multipliers from the basic rows: Bᵀ y = −(g + H dx + Hₙq dq)_B.
        y_new = -(transpose(sys.BF) \ _basic_gradient(g, H, dx, pb, dq, jb))
        # Shortened to the first amount, not negligible, the step takes below
        # its bound; set on their bounds, the others.
        β = clamp ? one(T) :
            minimum(((lb[i] - n[i]) / dx[i] for i in eachindex(n) if n[i] > max(lb[i], θ) && n[i] + dx[i] < lb[i]); init = one(T))
        n .= Base.clamp.(n .+ β .* dx, lb, ub)
        _advance_params!(q, β, dq, prob)
        y .+= β .* (y_new .- y)
    end
    return (; best..., converged, saddle, iterations)
end
