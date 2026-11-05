# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── active_set.jl ──────────────────────────────────────────────────────────────
#
# The iteration of Optima (reaktoro/optima, `MasterSolver`), as Reaktoro runs it
# for chemical equilibrium: an active set on the lower bounds, not a barrier
# schedule, with the changes `solve_active_set` lists, each measured against the
# iteration as published.

"""
    solve_active_set(prob, opts; u0 = nothing, y0 = nothing) -> OptimaResult

Minimize `f` subject to `A n = b` and `lb ≤ n ≤ ub` by the active-set Newton
iteration of Optima, the solver Reaktoro computes its equilibria with, from the
amounts `u0` and the multipliers `y0`. `opts.algorithm == :active_set` selects it
in [`solve`](@ref); the interior point stays available as `:interior_point`.

# One iteration

At the amounts `n` and the multipliers `y`, with `g = ∇f(n) − τ ./ n` and
`s = g + Aᵀy` the reduced gradient:

1. **Basis.** `m` basic variables are chosen by pivoted QR on `A diag(w)`,
   `wᵢ = nᵢ` (the smallest representable weight for a variable on its bound):
   the most abundant species carry the balance, as Optima weights its basis.
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

# What differs from Optima, and why

Each change was measured against the iteration as published, on the test suites
of OptimaSolver and ChemistryLab and on cement equilibria.

- **The barrier is intensive.** Reaktoro adds `−τ Σ ln nᵢ` with `τ = 1e-16 mol`:
  how much it moves the answer depends on how much matter the system holds. Here
  `τ = opts.barrier_rel · max|b|`, `1e-20` by default; at `1e-16` the barrier
  pulled a trace such as `H⁺` at pH 13 by 0.1 RT in its potential, and the cement
  equilibria took longer.
- **The balance is judged row by row,** relative to what the row holds, and a row
  without a budget (an element absent from the system) relative to the scale of
  the whole budget, as the dual Newton judges it; Optima's error is in moles.
- **A saddle point is not an answer.** On a non-convex Gibbs energy (a solid
  solution with a miscibility gap) Newton's method with the exact Hessian converges
  as readily to a saddle point as to a minimum, and Optima accepts every step. The
  reduced Hessian is tested at convergence; where it has a negative curvature, the
  solve is made again from the same start with Ipopt's inertia correction (the
  Hessian shifted by `δ diag(1/n)`, `δ` raised until the reduced Hessian is
  positive definite; Wächter and Biegler, Math. Program. 106 (2006) 25–57). Made
  at every iteration instead, the correction slowed the cement equilibria by a
  factor of four.

- **A negligible amount does not shorten the step.** Optima shortens the whole
  step to the first variable it would take below its bound; a species at 1e-15 mol
  then holds back every other, and a cement start that holds a hundred such traces
  took a hundred iterations to set them on their bounds one by one. A variable
  below `opts.negligible_rel · max|b|` (`1e-12` by default) is set on its bound
  instead, the balance it then misses restored by the next step, whose right-hand
  side holds it. At `1e-9` the steps lost too much matter, and the cement
  equilibria took longer again.
- **The start is not lifted.** The interior point lifts the absent species off
  their bounds, which it must; here a species on its bound is a variable the first
  iteration can declare unstable, as Reaktoro starts from the state as it is.
- **The Hessian may come from the caller.** A function `p.hess(H, n)` among the
  parameters fills the Hessian of `f` in place of its forward-mode Jacobian of
  the gradient, which costs as many gradient evaluations as there are variables,
  divided by the chunk. ChemistryLab hands over the exact diagonal and the ideal
  coupling within each phase, as Reaktoro approximates its Hessian. The saddle
  test at convergence reads the exact one.
- **A cycle ends.** On a Hessian that is not exact, the active set can cycle: a
  variable leaves its bound, the next step brings it back. A descent only lowers
  the error, and a run that leaves a saddle point or a poor first step raises it
  before lowering it; a cycle lowers it and raises it again. After ten such turns
  without halving its best error, the run ends on its best iterate.

As Reaktoro does, a solve that does not converge is made once more, its steps
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

_active_set_error(r) = max(r.err_x, r.err_w)

# The turns of its error back up after a fall that a run of the active set may
# take without halving its best error, beyond which it has stalled.
const ACTIVE_SET_STALL_WINDOW = 10

# The start, from a previous answer or its state as `solve` takes them, or from
# the amounts and multipliers given.
_as_start(u0::Union{OptimaResult, OptimaState}, y0) = (u0.n, u0.y)
_as_start(u0, y0) = (u0, y0)

_start_amounts(prob::OptimaProblem, ::Nothing) = _default_initial_n(prob)
_start_amounts(prob::OptimaProblem{T}, u0::AbstractVector) where {T} = clamp.(convert(Vector{T}, u0), prob.lb, prob.ub)

# Multipliers fitted to the stationarity of the most abundant species, or those
# given.
function _start_multipliers(prob::OptimaProblem, g, n, ::Nothing)
    big = partialsortperm(n, 1:min(prob.ns, max(3prob.m, prob.m + 4)); rev = true)
    return LinearAlgebra.qr(transpose(prob.A[:, big]), LinearAlgebra.ColumnNorm()) \ (-g[big])
end
function _start_multipliers(prob::OptimaProblem{T}, g, n, y0::AbstractVector) where {T}
    length(y0) == prob.m || throw(
        DimensionMismatch("`y0` holds $(length(y0)) multipliers for $(prob.m) conservation rows."),
    )
    return convert(Vector{T}, y0)
end

# The `m` basic variables of `A`, chosen by pivoted QR on `A diag(w)`: the columns
# of the largest weights first, as Optima's canonicalizer orders its basis; by
# pivoted QR on `A` itself where those columns are singular (weights so far apart
# that the order of the smallest is rounding).
function _weighted_basis(A::AbstractMatrix, w::AbstractVector)
    m = size(A, 1)
    jb = sort(LinearAlgebra.qr(A .* transpose(w), LinearAlgebra.ColumnNorm()).p[1:m])
    LinearAlgebra.issuccess(LinearAlgebra.lu(A[:, jb]; check = false)) && return jb
    return sort(LinearAlgebra.qr(A, LinearAlgebra.ColumnNorm()).p[1:m])
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

# The Hessian of `f − τ Σ ln nᵢ` at `n`: the caller's plus the barrier's, or the
# forward-mode Jacobian of the gradient.
function _hessian!(H, hess, grad!, gbuf, n, cfg, τ)
    hess(H, n)
    H[LinearAlgebra.diagind(H)] .+= τ ./ n .^ 2
    return H
end
_hessian!(H, ::Nothing, grad!, gbuf, n, cfg, τ) = ForwardDiff.jacobian!(H, grad!, gbuf, n, cfg)

_posdef(M) = LinearAlgebra.issuccess(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(M); check = false))

# One run of the iteration from `(u0, y0)`. `clamp`: the step is clamped into the
# bounds instead of shortened to the first one crossed (Reaktoro's second try).
# `inertia`: the reduced Hessian is shifted to positive definiteness at every
# step.
function _active_set_run(prob::OptimaProblem{T}, opts::OptimaOptions, u0, y0; clamp::Bool, inertia::Bool) where {T}
    ns, m = prob.ns, prob.m
    A, b, lb, ub = prob.A, prob.b, prob.lb, prob.ub
    n = _start_amounts(prob, u0)
    τ = T(opts.barrier_rel) * T(_system_scale(b))
    θ = T(opts.negligible_rel) * T(_system_scale(b))
    grad! = let prob = prob, τ = τ
        (gg, x) -> begin
            prob.g!(gg, x, prob.p)
            gg .-= τ ./ x
            return gg
        end
    end
    hess = _hessian_callback(prob.p)
    g = zeros(T, ns)
    gbuf = zeros(T, ns)
    H = zeros(T, ns, ns)
    cfg = ForwardDiff.JacobianConfig(grad!, gbuf, n, ForwardDiff.Chunk(n), ForwardDiff.Tag(grad!, eltype(n)))
    grad!(g, n)
    y = _start_multipliers(prob, g, n, y0)

    best = (; n = copy(n), y = copy(y), err_x = T(Inf), err_w = T(Inf), err_w_abs = T(Inf))
    converged = false
    saddle = false
    iterations = 0
    progress_ref, turns, err_prev, falling = T(Inf), 0, T(Inf), false
    for it in 1:(opts.max_iter)
        iterations = it
        grad!(g, n)
        s = g .+ transpose(A) * y
        at_lb = n .<= lb
        # weights on the values: a basis is a choice, not a function to
        # differentiate
        jb = _weighted_basis(_primal_value.(A), abs.(_primal_value.(n)))
        basic = falses(ns)
        basic[jb] .= true
        unstable = .!basic .& at_lb .& (s .> 0)
        ew = A * n .- b
        err_x = maximum(i -> (unstable[i] || (basic[i] && at_lb[i])) ? zero(T) : abs(s[i]), 1:ns)
        err_w = _balance_error(ew, A, b, n)
        if max(err_x, err_w) < max(best.err_x, best.err_w)
            best = (; n = copy(n), y = copy(y), err_x, err_w, err_w_abs = maximum(abs, ew))
        end
        # A search that cycles between active sets lowers its error, then
        # raises it again, over and over, without improving its best iterate.
        # A descent only lowers it, and one that leaves a saddle point or a
        # poor first step raises it before lowering it: neither turns back up
        # after a fall. `STALL_WINDOW` such turns without halving the best
        # error end the run on its best iterate.
        err = max(err_x, err_w)
        if max(best.err_x, best.err_w) < T(0.5) * progress_ref
            progress_ref, turns = max(best.err_x, best.err_w), 0
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
        if max(err_x, err_w) <= opts.tol
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
        dz = isempty(jn) ? T[] : LinearAlgebra.lu(Hr; check = false) \ sys.rhs
        all(isfinite, dz) || break
        dx = zeros(T, ns)
        dx[jn] .= dz
        dx[jb] .= sys.dxp .- sys.R * dz
        # Multipliers from the basic rows: Bᵀ y = −(g + H dx)_B.
        y_new = -(transpose(sys.BF) \ (g[jb] .+ (H * dx)[jb]))
        # Shortened to the first amount, not negligible, the step takes below
        # its bound; set on their bounds, the others.
        β = clamp ? one(T) :
            minimum(((lb[i] - n[i]) / dx[i] for i in eachindex(n) if n[i] > max(lb[i], θ) && n[i] + dx[i] < lb[i]); init = one(T))
        n .= Base.clamp.(n .+ β .* dx, lb, ub)
        y .+= β .* (y_new .- y)
    end
    return (; best..., converged, saddle, iterations)
end
