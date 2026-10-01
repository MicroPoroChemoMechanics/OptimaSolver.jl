# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── line_search.jl ─────────────────────────────────────────────────────────────
# Filter line search combining feasibility and objective decrease.
#
# Strategy (Wächter & Biegler 2006 style, simplified):
#   - Maintain a filter set {(θ, φ)} of (feasibility, objective) pairs.
#   - Accept a step α if (θ_new, φ_new) is not dominated by any filter entry
#     AND satisfies a sufficient-decrease condition.
#   - Backtrack by factor β if rejected.

"""
    FilterEntry{T}

One entry in the line-search filter: (feasibility norm, objective value).
"""
struct FilterEntry{T <: Real}
    θ::T    # ‖An - b‖₁  (feasibility measure)
    φ::T    # f(n)        (objective)
end

dominates(e1::FilterEntry, e2::FilterEntry) = e1.θ <= e2.θ && e1.φ <= e2.φ

"""
    LineSearchFilter{T}

Mutable filter for the line search.
"""
mutable struct LineSearchFilter{T <: Real}
    entries::Vector{FilterEntry{T}}
end

LineSearchFilter(T::Type = Float64) = LineSearchFilter(FilterEntry{T}[])

function is_acceptable(f::LineSearchFilter{T}, θ::T, φ::T) where {T}
    candidate = FilterEntry{T}(θ, φ)
    return !any(e -> dominates(e, candidate), f.entries)
end

function add_to_filter!(f::LineSearchFilter{T}, θ::T, φ::T) where {T}
    push!(f.entries, FilterEntry{T}(θ, φ))
    return f
end

"""
    line_search(prob, n, y, dn, dy, f_val, grad_f, μ, opts; filter) -> (α, n_new, y_new, f_new)

Backtracking line search with filter acceptance.

Starting from α = α_max (from fraction-to-boundary), tries α, β*α, β²*α, …
until the new point (n + α dn, y + α dy) is accepted by the filter or the
Armijo condition on feasibility is satisfied.

Returns the accepted step size α and the new iterates.
"""
function line_search(
        prob::OptimaProblem{T},
        n::AbstractVector,
        y::AbstractVector,
        dn::AbstractVector,
        dy::AbstractVector,
        f_val,
        grad_f::AbstractVector,
        μ,
        opts::OptimaOptions;
        filter::LineSearchFilter,
        α_max = 1.0,
        kkt_merit = nothing,
    ) where {T}
    # `α_max` comes from `clamp_step`, hence carries the type of the iterates:
    # annotating it `::Float64` used to reject dual numbers outright, which is
    # what stopped `ForwardDiff` from crossing the solve.
    Tv = promote_type(eltype(n), eltype(dn), typeof(μ), typeof(α_max))

    α = Tv(α_max)
    β = Tv(opts.ls_beta)

    # Current feasibility, row by row and in sum
    r_curr = prob.A * n .- prob.b
    θ_curr = sum(abs, r_curr)

    # Barrier objective at current point: f_μ(n) = f(n) - μ Σ ln(nᵢ - lbᵢ)
    # A slack that has rounded to zero or below is a caller error, but it must not
    # be a `DomainError` from inside `log`: the barrier is floored so the search
    # can still report a step, and the point is rejected on its merits.
    s_curr = max.(n .- prob.lb, floatmin(T))
    f_μ_val = real(f_val) - μ * sum(log, s_curr)

    # Directional derivative of the BARRIER objective along dn:
    #   ∇f_μ · dn = (∇f - μ/s) · dn
    # This is guaranteed ≤ 0 for the IPM Newton step (H is positive definite).
    descent_μ = dot(grad_f .- μ ./ s_curr, dn)

    # When already feasible (θ_curr ≈ 0) the filter entry (0, f_old) blocks
    # any step that does not strictly decrease f, even if f_μ decreases.
    # In that regime, bypass the filter and rely on Armijo on f_μ alone.
    θ_tol = sqrt(eps(T)) * max(one(T), θ_curr)
    use_filter = θ_curr > θ_tol

    # !!! note "Armijo below the resolution of `f`, and why it is left alone"
    #     Both sides of the test below are numbers of the size of `f_μ`. Once the
    #     decrease being asked for, `c α |∇f_μ·dn|`, falls under the spacing of
    #     floating-point numbers near `f_μ`, the comparison is settled by rounding
    #     rather than by the function, and backtracking halves `α` at random.
    #     Measured on the three-species ideal solution, iterations 22 through 39
    #     returned `α = 0.0039`, `0.0078`, then `1.91e-6` four times running while
    #     `‖dn‖` sat at `4e-12`; the KKT error took those seventeen iterations to
    #     crawl from 5.1e-11 to 6.5e-12. The noise is in `f` itself — the barrier
    #     term contributes `1e-11` against `f ≈ 0.6`, so nothing inside this
    #     function can be reformulated to recover the digits.
    #
    #     Two remedies were implemented and both measured worse, so neither is here:
    #
    #     - Taking `α_max` whole when the predicted decrease is below the noise
    #       floor. It cut the toy problem from 55 iterations to 32 at equal
    #       accuracy, but a step whose merit cannot be assessed is a gamble, and on
    #       the Reaktoro coupling reference it moved the `t = 0` speciation of pure
    #       water from 0.9 % to 18.8 % wrong on OH⁻.
    #     - Treating the same condition as "this barrier level is finished" and
    #       reducing `μ`. That is not a proximity test at all: `∇f_μ·dn` is small
    #       whenever the curvature `μ/sᵢ²` is large, which happens far from the
    #       solution too, so the barrier raced to its floor and the solver returned
    #       an answer wrong at `1.3e-6` without converging.
    #
    #     Ipopt's own tiny-step check (`IpoptAlgorithm::CheckTinyStep`) tests
    #     `maxᵢ|dnᵢ|/max(1,|nᵢ|)` against `10·eps`, which is `6e-12` against
    #     `2.2e-15` here: faithful Ipopt does nothing in this regime either.
    #
    #     What is done instead, since 0.7.5, is to judge the step on a quantity
    #     that is resolved there: the KKT residual of the barrier problem in
    #     complementarity form, `‖s∘(∇f + Aᵀy) − μ‖` with the balance (`kkt_merit`),
    #     which is bounded and measured at its own scale, `1e-11` to `1e-17`
    #     rather than `1e-11` within `0.6`. A step is still refused unless it
    #     lowers that residual by Armijo's fraction, so nothing is taken on
    #     trust. Left to rounding, the same three-species solve stalled at
    #     `MaxIters` on a continuous-integration machine whose arithmetic differed
    #     in the last digits from the one it converged on in 44 iterations.
    #
    #     Only there, and only for a step that the objective cannot tell apart from
    #     no change: one that raises `f_μ` by more than its rounding, or moves the
    #     balance, is refused as before. Taken whenever Armijo was unresolved, the
    #     residual let through steps far from the solution, where the curvature
    #     makes the decrease asked for small too, and moved the interior-point
    #     answer of the Reaktoro reference by half on a trace.
    #
    #     Resolvability is judged at each step tried, not at the first one only:
    #     backtracking from a resolvable step comes down into the same regime.
    noise_f = 8 * length(n) * eps(T) * max(abs(f_μ_val), one(T))
    R_curr = nothing          # the residual at `(n, y)`, computed when first needed

    for _ in 1:(opts.ls_max_iter)
        n_new = n .+ α .* dn
        y_new = y .+ α .* dy

        # Enforce positivity
        if any(i -> n_new[i] <= prob.lb[i], eachindex(n_new))
            α *= β
            continue
        end

        f_new = prob.f(n_new, prob.p)
        r_new = prob.A * n_new .- prob.b
        θ_new = sum(abs, r_new)
        s_new = max.(n_new .- prob.lb, floatmin(T))
        f_μ_new = real(f_new) - μ * sum(log, s_new)

        if use_filter
            # Filter: stores (θ, f) — per Wächter & Biegler 2006
            if !is_acceptable(filter, Tv(θ_new), Tv(real(f_new)))
                α *= β
                continue
            end
            # Accept if feasibility decreases OR barrier objective satisfies Armijo
            if θ_new <= θ_curr * (one(Tv) - Tv(opts.ls_alpha)) ||
                    f_μ_new <= f_μ_val + opts.ls_alpha * α * descent_μ
                return α, n_new, y_new, f_new
            end
        else
            # Already feasible: pure Armijo on barrier objective
            if f_μ_new <= f_μ_val + opts.ls_alpha * α * descent_μ
                return α, n_new, y_new, f_new
            end
            # The barrier objective no longer resolves the decrease asked for, and
            # this step changes it by no more than its rounding: judged on the
            # residual of the optimality conditions, the balance kept.
            if kkt_merit !== nothing && opts.ls_alpha * α * abs(descent_μ) <= noise_f &&
                    f_μ_new <= f_μ_val + noise_f && _balance_kept(prob, r_new, r_curr, n_new, sqrt(eps(T)))
                R_curr === nothing && (R_curr = kkt_merit(n, y))
                kkt_merit(n_new, y_new) <= (one(Tv) - Tv(opts.ls_alpha) * α) * R_curr &&
                    return α, n_new, y_new, f_new
            end
        end

        α *= β
    end

    # Fallback: return smallest tried step (better than nothing)
    n_new = n .+ α .* dn
    n_new .= max.(n_new, prob.lb .+ eps(T))
    return α, n_new, y .+ α .* dy, prob.f(n_new, prob.p)
end

# Whether a step keeps the balance: every row no worse than it was, up to `√eps`
# of that row's own scale, `|bₖ| + Σⱼ |Aₖⱼ nⱼ|`.
#
# Row by row, because the rows do not share a scale. Judged on the sum against
# `√eps` absolute, as 0.7.5 did, a trace component could vanish whole inside a
# tolerance the major elements set. Measured on blended cement pastes carrying a
# trace of carbon nine orders of magnitude below their major elements: the
# interior point lost it, every start of the certified search downstream came
# from there, and the answer was stationary to rounding with every carrier of
# carbon orders of magnitude too low, the carbon balance wrong by its whole
# budget. The major rows keep the latitude they had.
function _balance_kept(prob, r_new, r_curr, n_new, tol)
    for k in eachindex(r_new)
        scale = abs(prob.b[k])
        for j in eachindex(n_new)
            scale += abs(prob.A[k, j] * n_new[j])
        end
        abs(r_new[k]) <= max(abs(r_curr[k]), tol * scale) || return false
    end
    return true
end

# The KKT residual of the barrier problem at `(n, y)`, in complementarity form,
# `‖s∘(∇f + Aᵀy) − μ‖` together with the balance `‖An − b‖`, `g` a buffer for the
# gradient. Bounded as a bound is approached, and resolved at its own scale.
function _kkt_merit(prob, n, y, g, μ)
    eval_gradient!(g, prob, n)
    s = n .- prob.lb
    gL = g .+ prob.A' * y
    return sqrt(sum(abs2, s .* gL .- μ) + sum(abs2, prob.A * n .- prob.b))
end
