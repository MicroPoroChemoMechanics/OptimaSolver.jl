# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── stability.jl ───────────────────────────────────────────────────────────────
# Variable stability classification.
#
# A variable nᵢ is "unstable" if it is close to its lower bound lb[i] and its
# reduced cost (stationarity residual) pushes it further toward the bound.
# Unstable variables get a smaller step fraction to avoid numerical issues
# near zero.

"""
    stability_measure(n, lb, ex) -> Vector

Compute the stability measure sᵢ = (nᵢ - lbᵢ) * |exᵢ|.

Small sᵢ means variable i is both close to its bound AND has large stationarity
residual — likely to be at an active bound at the solution.
"""
function stability_measure(n::AbstractVector, lb::AbstractVector, ex::AbstractVector)
    return (n .- lb) .* abs.(ex)
end

"""
    classify_variables(n, lb, ex; tol_stable=1e-8) -> (js, ju)

Return indices of stable (`js`) and unstable (`ju`) variables.

A variable is stable if (nᵢ - lbᵢ) > tol_stable * max_slack.
"""
function classify_variables(
        n::AbstractVector,
        lb::AbstractVector,
        ex::AbstractVector;
        tol_stable::Float64 = 1.0e-8,
    )
    slack = n .- lb
    max_slack = maximum(slack)
    threshold = tol_stable * max_slack

    # A variable is unstable if it is both near its lower bound AND its
    # stationarity residual points toward the bound (ex[i] > 0 means ∂G/∂nᵢ
    # pushes nᵢ downward when the multiplier contribution is small).
    js = findall(i -> slack[i] > threshold || ex[i] < zero(eltype(ex)), eachindex(n))
    ju = findall(i -> slack[i] <= threshold && ex[i] >= zero(eltype(ex)), eachindex(n))
    return js, ju
end

"""
    reduced_step_for_unstable!(dn, ju, n, lb; τ=0.5)

For unstable variables (near lower bound), cap the step to move at most
τ * slack toward the bound to prevent crossing.
"""
function reduced_step_for_unstable!(
        dn::AbstractVector,
        ju::AbstractVector{Int},
        n::AbstractVector,
        lb::AbstractVector;
        τ::Float64 = 0.5,
    )
    for i in ju
        slack = n[i] - lb[i]
        if dn[i] < zero(eltype(dn))
            max_step = -τ * slack
            if dn[i] < max_step
                dn[i] = max_step
            end
        end
    end
    return dn
end
