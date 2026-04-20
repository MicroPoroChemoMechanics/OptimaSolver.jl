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
        α_max::Float64 = 1.0,
    ) where {T}
    Tv = promote_type(eltype(n), eltype(dn), typeof(μ))

    α = Tv(α_max)
    β = Tv(opts.ls_beta)

    # Current feasibility
    θ_curr = sum(abs, prob.A * n .- prob.b)

    # Barrier objective at current point: f_μ(n) = f(n) - μ Σ ln(nᵢ - lbᵢ)
    s_curr = n .- prob.lb
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

    for _ in 1:(opts.ls_max_iter)
        n_new = n .+ α .* dn
        y_new = y .+ α .* dy

        # Enforce positivity
        if any(i -> n_new[i] <= prob.lb[i], eachindex(n_new))
            α *= β
            continue
        end

        f_new = prob.f(n_new, prob.p)
        θ_new = sum(abs, prob.A * n_new .- prob.b)
        s_new = n_new .- prob.lb
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
        end

        α *= β
    end

    # Fallback: return smallest tried step (better than nothing)
    n_new = n .+ α .* dn
    n_new .= max.(n_new, prob.lb .+ eps(T))
    return α, n_new, y .+ α .* dy, prob.f(n_new, prob.p)
end
