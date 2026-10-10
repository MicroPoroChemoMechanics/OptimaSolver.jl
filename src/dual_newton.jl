# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy (Cerema, UMR MCD)

# ── dual_newton.jl ────────────────────────────────────────────────────────────
#
# Exact Newton on the KKT system, in the space of the equality multipliers.
#
# THE PROBLEM. Minimize `f(x) = Σᵢ xᵢ(gᵢ + hᵢ(x))` subject to `A x = b`, `x ≥ 0`,
# where `h` is a callback supplying the state-dependent part of the gradient and
# `∇f = g + h(x)` (this identity holds whenever `Σⱼ xⱼ ∂hⱼ/∂xᵢ = 0`, which is
# the Gibbs–Duhem relation of a chemical system and the Euler relation of any
# first-order homogeneous mixing term).
#
# It is convex whenever `h` is the gradient of a convex mixing term and the
# remaining variables enter linearly, so the KKT conditions are necessary AND
# sufficient: a point satisfying them is proved globally optimal.
#
# THE FORMULATION. With `y` the multipliers of `A x = b` and `u := −Aᵀy`, the
# stationarity conditions split by the nature of the variable:
#
#   interior variable   hᵢ(x) = uᵢ − gᵢ           (invertible: a mass-action law)
#   bounded variable    gᵢ = uᵢ if present, gᵢ ≥ uᵢ if at its bound
#
# The second line is a phase-stability criterion: a bounded variable is positive
# exactly when its "saturation index" `uᵢ − gᵢ` vanishes, and zero when negative.
#
# WHY IT IS WELL CONDITIONED. Parameterizing the interior variables by `w = ln x`
# makes their positivity automatic, so no fraction-to-boundary rule applies to
# them — and that rule is what caps the step of the interior-point method in
# `solve!` at every single iteration. Only the bounded variables carry a bound,
# and they are handled by an active set, exactly and finitely.
#
# This is NOT the log reparameterization of the objective that
# `variable_space = Val(:log)` performs. `f ∘ exp` is not convex — its second
# derivative `xᵢ(∇fᵢ + 1)` is negative wherever `∇fᵢ < −1`, which for a chemical
# potential of order −200 is everywhere. Here the logarithm is applied to the KKT
# EQUATIONS, solved as a square nonlinear system; convexity of the original
# problem is what makes that system's solution unique.

using LinearAlgebra

"""
    DEGENERATE_POTENTIAL

Multiplier assigned to a component whose right-hand side has vanished. Any value
large enough to drive `exp(uᵢ − gᵢ)` below the floor for every variable carrying
that component will do; the solution does not depend on it.
"""
const DEGENERATE_POTENTIAL = 500.0

# Bounds on a log-molality. The floor is where `exp` underflows; the ceiling is
# far above any physical molality and exists only to keep a diverging iterate
# finite. Both are ACTIVE BOUNDS, not tolerances: see `_invert_phases!`.
const W_FLOOR = -700.0

# The floor of the vertex of the linear program as a start (`lp_fallback` of
# `DualNewtonOptions`): what the vertex leaves at zero starts there.
const LP_FALLBACK_FLOOR = 1.0e-16
const W_CEIL = 20.0

# Sweeps without a new smallest step after which `_invert_phases!` stops.
#
# The inner fixed point converges linearly when it converges, so its step sets a
# new minimum at every sweep; it only stops doing so when it has stopped
# converging. Measured on a 107-species cement solved cold, 89 % of the 69 463
# calls ran to the 200-sweep cap, and they were not creeping towards the
# threshold: their final step had a median of 30, the bound on a single update,
# and their traces read 30, 28, 30, 28 ... with the step at sweep 200 equal to
# the step at sweep 100. A cycle bounded by the clamp, at trial points whose
# potentials no composition within the bounds satisfies. Each of those calls
# paid 200 sweeps to return a state the line search then refused.
#
# Stopping them costs nothing an accepted step depended on, since they end above
# `inner_tol` either way. With the value below, the same solve spends about 21
# sweeps on each such call instead of 200: 1.40 million sweeps in all against
# 12.8 million, and 16.8 s against 142.6 s, certified and returning the same
# composition to 4e-13.
#
# What this does to the outer path is not fixed by the rule. The state returned
# at a stalled call differs from the one the two-hundredth sweep would have
# returned, and the outer Newton follows it: at a window of 10 the same solve
# needed 492 calls and certified on its first route in 0.2 s, at 25 it needed
# 28 150 and took 20.5 s, at 20 it needs 63 661. The composition was the same to
# 3e-10 in every case. The value is set by the argument above, well past the
# period of the cycles observed, and not by the fastest of those runs.
const INNER_STALL_SWEEPS = 20

# The most a solute's log-amount may rise in one sweep of `_invert_phases!`.
#
# A rise is what can overshoot: a solute whose stationarity asks for thirteen
# orders of magnitude more, granted in one sweep, can carry the ionic strength
# from 0.1 to 100 mol/kg, and the next sweep swings back. How far it may FALL is
# `DualNewtonOptions.inner_fall_bound`, the same 30 by default.
const W_MAX_RISE = 30.0

# The amount a mixing phase is admitted with, as its total.
#
# It has to survive the first Newton step. Admitted at 1e-9, a phase could
# never be kept: the step changes `ln N` by at most one, so after it the total
# is at most 2.7e-9, below `si_tol` (1e-8), the Newton iteration of the round
# stops on it and the round drops it as vanished — to be admitted again at the
# next round, with the same outcome, until the set repeats and the search ends
# with the phase at e·1e-9.
# A mixing phase could therefore be present only if the starting point already
# held it; from a start without it, a supersaturated solid solution stayed out.
# The seeding of the initial active set lifts a phase total to the same value,
# for the same reason.
const PHASE_ADMISSION_SEED = 1.0e-6

"""
    _degenerate_conservation_rows(prob, b) -> Vector{Int}

The degenerate rows of `prob`, restricted to those the criterion applies to.

`degenerate_components` answers a question about element conservation; asking it
of a reactivity row gets the wrong answer, so `prob.conservation_rows` says which
rows it may be asked about.
"""
function _degenerate_conservation_rows(prob, b::AbstractVector)
    rows = prob.conservation_rows
    length(rows) == size(prob.A, 1) && return degenerate_components(prob.A, b)
    isempty(rows) && return Int[]
    sub = degenerate_components(prob.A[rows, :], b[rows])
    return [rows[k] for k in sub]
end

"""
    degenerate_components(A, b) -> Vector{Int}

Rows `k` of `A x = b` whose right-hand side forces every variable carrying
component `k` to zero.

The criterion is **not** simply `b_k ≈ 0`. With `x ≥ 0`, the row
`Σᵢ A_{ki} xᵢ = 0` forces `xᵢ = 0` for all `i` with `A_{ki} ≠ 0` only when the
non-zero entries of the row share a sign: a sum of non-negative terms vanishes
only if each vanishes. A row with entries of both signs permits cancellation and
forces nothing.

That distinction is not academic. In a chemical system the `H+` row carries `−1`
for `OH-` and `+1` for `H+`, so `b = 0` there is the ordinary state of pure
water — declaring it degenerate kills the whole acid–base system and returns
pH 7.000 with the solid undissolved.

The signs are read over the variables still free, so the answer is a fixed
point: a row found degenerate forces its variables to zero, and a row whose
entries of the other sign all sat on those variables forces its own in turn. The
electron row of a system with sulfate and chloride is one. Its budget is zero, it
carries one sign on the reduced sulfur species and the other on perchlorate, so
alone it forces nothing; with no chlorine in the budget, perchlorate is zero, and
nothing is left to take the electrons of a sulfide. Read over every column, the
row was left free: its multiplier then has to run to infinity for the sulfide to
vanish, and the Newton iteration on a carbonated cement paste stagnated there,
while the same paste without its two chloride phases solved at once.
"""
function degenerate_components(A::AbstractMatrix, b::AbstractVector)
    m = size(A, 1)
    scale = max(maximum(abs, b; init = 0.0), 1.0)
    out = Int[]
    forced = falses(size(A, 2))
    changed = true
    while changed
        changed = false
        @inbounds for k in 1:m
            (abs(b[k]) <= 1.0e-12 * scale && !(k in out)) || continue
            pos = false
            neg = false
            for j in axes(A, 2)
                forced[j] && continue
                a = A[k, j]
                a > 0 && (pos = true)
                a < 0 && (neg = true)
                pos && neg && break
            end
            (pos && neg) && continue
            push!(out, k)
            changed = true
            for j in axes(A, 2)
                iszero(A[k, j]) || (forced[j] = true)
            end
        end
    end
    return sort!(out)
end

"""
    SolutionPhase(members, j_ref; always_present = false)

A set of variables whose `hᵢ` depends on the composition of the set — a mixing
phase — together with the position **within `members`** of its reference.

Every member of such a phase satisfies `hᵢ = uᵢ − gᵢ`, and for all but the
reference that condition is inverted directly. The reference is exempt for a
structural reason: `hᵢ` is a logarithm of a mole fraction, hence **bounded
above**, so `hᵢ = uᵢ − gᵢ` has no solution for an arbitrary `y`. Inverting it is
not merely slow, it can be infeasible, and an inner loop that included it can
never report convergence — which invalidates the outer Jacobian, that Jacobian
being derived on the assumption the inner conditions hold exactly. Its
stationarity is carried by the outer system, where it fixes the phase's total.

`always_present = true` exempts the phase from the stability test: the aqueous
phase of a wet system is there by hypothesis, a solid solution is not.
"""
struct SolutionPhase
    members::Vector{Int}
    j_ref::Int
    always_present::Bool
    mole_fraction::Bool
    split_starts::Vector{Vector{Float64}}
    newton::Bool
    bounded_members::Vector{Int}
    local_h::Union{Nothing, Function}
    invert::Union{Nothing, Function}
end

"""
    SolutionPhase(members, j_ref; always_present = false, mole_fraction = false,
                  split_starts = Vector{Float64}[], newton = false,
                  bounded_members = Int[], local_h = nothing, invert = nothing)

One mixing phase: the variable indices it holds, which of them is the reference,
whether it may leave, whether EVERY member's activity is a mole fraction, and
where to look when asking whether it wants to unmix.

`split_starts` is that last one: extra trial compositions for
[`phase_split_trial`](@ref), as mole fractions over `members`. It is empty by
default and the search then probes the corners of the simplex alone, which finds
a phase that is **unstable** but misses one that is merely **metastable** —
outside its spinodal, inside its binodal — where every corner start walks back to
the phase's own composition. A caller that knows the mixing model can compute its
binodal directly and hand it over here; the search then refines it with the
chemical potentials the full system has. This solver knows the system and not the
model, so the model's half has to come from outside.

That last flag decides how the members are recovered, and the two cases are
genuinely different rather than a matter of taste.

In an aqueous solution the solutes carry molalities, unbounded above, so `hᵢ` can
meet `uᵢ − gᵢ` whatever its value and each member is recovered from its own
stationarity. Only the solvent is a mole fraction, which is exactly why it is the
reference and why its equation is carried by the outer system instead.

In a solid solution there is no solvent: every member is a mole fraction and
every `hᵢ` is bounded above by zero. Asking any of them to meet a positive
`uᵢ − gᵢ` is asking for the impossible, and the iteration answers by growing the
member without bound. Such a phase is recovered from the RATIOS between its
members, which are always attainable, with the absolute level left to the
reference's own equation — where it belongs.

# Recovering the composition: substitution or Newton

By default the ratios are found by successive substitution,
`x = N · softmax(u − g − lnγ(x))`. Written in `w = ln x`, one sweep is
`w ← w + (u − g − h(w))` followed by a renormalization, so its Jacobian is
`I − ∂h/∂w` on the simplex. For an ideal phase `∂h/∂w` is the projector that
removes the total, the map is exact in one sweep, and a weak excess term keeps
it a contraction. A model whose `∂h/∂w` has eigenvalues above two does not: the
map then diverges. Ideal mixing on sublattices can be such a model: there the
eigenvalues are bounded by the sum of the site multiplicities, which is nine for
the CNASH gel of Myers et al., instead of one. Measured on that gel, at the
potentials of a CEM I paste, the substitution diverged within six sweeps.

`newton = true` recovers the composition instead by Newton's method on the
bordered system `[H 1; xᵀ 0]`, `H = ∂h/∂w` over the members, obtained by
`ForwardDiff`: `h` must then accept dual numbers. The same iteration serves the
tangent-plane test of the phase, whose stationary points solve the same
equations. The defaults leave every existing phase exactly as it was.

`local_h`, a function of the members' amounts alone (a vector over `members`)
returning their `h`, may be given when those depend on nothing else, as a solid
solution's do. The Newton iteration then differentiates it rather than the whole
`h`, which for a cement means eight variables instead of a hundred, and must
return what `h` would return for them.

`invert`, for a phase with a solvent, recovers its solutes in one call instead of
by sweeps: `invert(c, ref, w, q, params)`, with `c` the `uᵢ − gᵢ` of the members
(a vector over `members`, `-Inf` for a member whose component is absent from the
budget), `ref` the amount of the reference, `w` their current log-amounts and
`q`, `params` what `h` is given, returns the log-amounts at which every member
but the reference meets `hᵢ = cᵢ`, or `nothing` when no composition does. A trial
of the line search it answers `nothing` is passed over when the iterate it comes
from had a composition, and swept as that iterate was when it had none. The sweep
below assumes `∂hᵢ/∂wᵢ = 1` and nothing else, and an activity model couples the
solutes: through the ionic strength, the Debye–Hückel and B-dot models give
`∂h/∂w` a part of rank one. Where that coupling is strong the
sweep cycles instead of converging, and where the model has no solution it
cycles as well, so the two cannot be told apart. A model whose coefficients
depend on the composition through the ionic strength alone reduces the
inversion to one equation in it, which a caller that knows the model solves
exactly, or proves has no root; that is how PHREEQC treats the ionic strength,
as an unknown of its own. The sweep remains for a phase without `invert`.

`bounded_members` lists the members (positions within `members`) that may be
**exactly absent from the phase while it is present**. A member's activity
normally vanishes with its fraction, so its stationarity is an equality however
small it is. A member that carries no species of its own on any site of a
sublattice model keeps a finite activity as it disappears: it can be absent,
and its condition is then an inequality, as a pure phase's is. The certificate
tests a bounded member below its floor by that inequality instead of excluding
it.
"""
function SolutionPhase(
        members, j_ref; always_present::Bool = false, mole_fraction::Bool = false,
        split_starts = Vector{Vector{Float64}}(), newton::Bool = false,
        bounded_members = Int[], local_h::Union{Nothing, Function} = nothing,
        invert::Union{Nothing, Function} = nothing,
    )
    invert !== nothing && mole_fraction && throw(
        ArgumentError(
            "SolutionPhase: `invert` recovers the solutes of a phase with a solvent; " *
                "a phase whose members are all mole fractions has no solvent to fix them by."
        )
    )
    newton && !mole_fraction && throw(
        ArgumentError(
            "SolutionPhase: `newton = true` inverts a phase whose members are all " *
                "mole fractions; a phase with a solvent recovers its solutes one by one."
        )
    )
    bm = collect(Int, bounded_members)
    all(in(eachindex(members)), bm) || throw(
        ArgumentError(
            "SolutionPhase: `bounded_members` are positions within `members` " *
                "(1:$(length(members))); got $bm."
        )
    )
    return SolutionPhase(
        collect(Int, members), j_ref, always_present, mole_fraction,
        # Starting points of a search, which decides on values: a binodal
        # computed from a model being differentiated arrives as dual numbers,
        # and only their values are a place to start from.
        Vector{Float64}[Float64[_primal_value(x) for x in s] for s in split_starts],
        newton, bm, local_h, invert,
    )
end

"""
    DualNewtonProblem(A, g, h; phases, idx_bounded = Int[], params = nothing,
                      gq = nothing, cq = nothing, hq = nothing, q0 = Float64[],
                      qscale = Float64[], Aq, always_active = Int[],
                      conservation_rows = 1:size(A, 1))

A convex program in the form solved by [`dual_newton_solve`](@ref):

```math
\\min_x \\sum_i x_i\\bigl(g_i + h_i(x)\\bigr)
\\quad\\text{s.t.}\\quad A x = b, \\; x \\ge 0 .
```

# Arguments

  - `A`: the `m × n` equality matrix.
  - `g`: the constant part of the gradient, `∇f = g + h(x)`.
  - `h`: callback `h(x, params) -> Vector`, the state-dependent part.

# Keywords

  - `phases`: the mixing phases, as [`SolutionPhase`](@ref) values. Their members
    are strictly positive while the phase exists, hence parameterized by `ln x`
    and recovered from their own stationarity.
  - `idx_bounded`: variables outside every phase. Their `hᵢ` does not depend on
    the composition — a pure phase, of unit activity — so they are either at a
    stationarity of their own or at zero, and an active set decides which.
  - `params`: passed through to `h`.
  - `q0`, `gq`, `cq`, `hq`, `Aq`: unknown parameters solved for together with
    the composition — a prescribed property (a pH, a temperature) or a reaction
    extent. `q0` is their starting guess, `cq(x, q, params)` their residual
    equations (one per parameter), `gq(q, params)` the standard part of the
    gradient at `q` when it depends on them, `hq(x, q, params)` the state-
    dependent part when it does, and `Aq` (`m × length(q0)`) puts them in the
    linear rows, which then read `A x + Aq q = b`.
  - `qscale`: accepted for compatibility and checked; the Jacobian is exact and
    reads no scale.
  - `conservation_rows`: which rows of `A x + Aq q = b` the degeneracy criterion
    of [`degenerate_components`](@ref) may be applied to. Defaults to all of them,
    which is right when every row conserves an element or the charge.

    It must NOT be all of them once a row means something else. A **reactivity**
    row `nᵢ − Σⱼ νᵢⱼ Δξⱼ = nᵢ(0)` has a single positive entry on `x` and, for a
    product that starts absent, a zero right-hand side — exactly the shape the
    criterion reads as "no matter of this component exists". It then pins that
    row's multiplier to `DEGENERATE_POTENTIAL` and declares the species dead, so a
    solid product can never form: measured, the stationarity residual sat at 458
    and the reaction extents came out 4.3 times short. "This species starts at
    zero" and "this component is absent from the system" are different statements,
    and only the second licenses the criterion.
  - `always_active`: bounded variables that are never dropped from the active set.
    A variable whose amount is fixed by a linear row is not deciding anything by a
    sign test: it is determined, and its extra multiplier makes its stationarity
    satisfiable at whatever amount the row demands. Without this it can never
    enter, because it starts at zero and the drop rule removes anything below
    `si_tol` — which is exactly what happens to the products of a solid-to-solid
    reaction whose extents pin them.

# The two kinds of variable

The distinction is not cosmetic. A pure phase satisfies `gᵢ + hᵢ = uᵢ` when
present and `≥` when absent, and it can be **exactly** zero. A member of a mixing
phase cannot: its activity goes to `−∞` as its fraction goes to zero, so it is
never exactly absent while the phase exists. The active set for a mixing phase is
therefore over the PHASE, and the criterion is a tangent-plane test rather than a
sign of a saturation index.
"""
struct DualNewtonProblem{T <: Real, H, G, C, HQ, P}
    A::Matrix{T}
    g::Vector{T}
    h::H
    phases::Vector{SolutionPhase}
    idx_bounded::Vector{Int}
    # Concrete, so that every `h(x, params)` of the sweeps is a static call
    # whose result has a known type: held as `Any`, it made each one dynamic
    # and the vector it returns of no inferable type.
    params::P
    # ── the q block: prescribed properties, solved for simultaneously ────────
    nq::Int              # number of unknown parameters (0 for a plain T, P solve)
    gq::G                # gq(q, params) -> Vector, the standard part at those q
    cq::C                # cq(x, q, params) -> Vector of length nq, the residuals
    hq::HQ               # hq(x, q, params) -> Vector, `h` when it depends on q
    Aq::Matrix{T}        # m × nq, so the linear rows read `A x + Aq q = b`
    always_active::Vector{Int}   # bounded variables pinned by a linear row
    conservation_rows::Vector{Int}   # rows the degeneracy criterion applies to
    q0::Vector{T}        # starting guess
    qscale::Vector{T}    # scale of each parameter (not read since the Jacobian is exact)
end

function DualNewtonProblem(
        A::AbstractMatrix, g::AbstractVector, h;
        phases::AbstractVector{SolutionPhase},
        idx_bounded::AbstractVector{Int} = Int[],
        params = nothing,
        gq = nothing,
        cq = nothing,
        hq = nothing,
        q0::AbstractVector = Float64[],
        qscale::AbstractVector = Float64[],
        Aq::AbstractMatrix = zeros(Float64, size(A, 1), length(q0)),
        always_active::AbstractVector{Int} = Int[],
        conservation_rows::AbstractVector{Int} = 1:size(A, 1),
    )
    isempty(phases) && throw(
        ArgumentError(
            "`phases` is empty: the multipliers are determined by the stationarity " *
                "of the variables inside a mixing phase, and without one there is " *
                "nothing to determine them."
        )
    )
    for (k, ph) in pairs(phases)
        1 <= ph.j_ref <= length(ph.members) || throw(
            ArgumentError("phase $k: `j_ref` = $(ph.j_ref) is not a position in `members`.")
        )
    end
    seen = Set{Int}()
    for ph in phases, i in ph.members
        i in seen && throw(ArgumentError("variable $i belongs to more than one phase."))
        push!(seen, i)
    end
    for i in idx_bounded
        i in seen && throw(
            ArgumentError("variable $i is both in a phase and bound-constrained.")
        )
    end

    nq = length(q0)
    if nq > 0
        # `gq` is optional. A prescribed temperature moves the standard
        # potentials, so it needs one; a reaction extent does not — it enters
        # through `Aq` and `cq` only, `g` staying put. Requiring `gq` refused a
        # perfectly well-posed kinetic step.
        cq === nothing && throw(
            ArgumentError(
                "`q0` has $nq entries but no `cq`: each unknown parameter " *
                    "needs one residual equation."
            )
        )
        # The Jacobian is exact and reads no `qscale`; one given is still
        # checked, so that no caller breaks, and none is needed.
        isempty(qscale) && (qscale = ones(nq))
        length(qscale) == nq || throw(
            ArgumentError(
                "`qscale` must have one entry per unknown parameter " *
                    "($nq), got $(length(qscale))."
            )
        )
        all(>(0), qscale) || throw(ArgumentError("`qscale` entries must be positive."))
        size(Aq) == (size(A, 1), nq) || throw(
            ArgumentError(
                "`Aq` must be $(size(A, 1))×$nq to sit beside `A` in the " *
                    "linear rows, got $(size(Aq))."
            )
        )
    end

    # The element type of the data is kept: a problem whose potentials carry
    # dual numbers is differentiated through (`_dual_newton_solve_ad`).
    T = promote_type(Float64, eltype(A), eltype(g), eltype(Aq), eltype(q0), eltype(qscale))
    return DualNewtonProblem(
        Matrix{T}(A), Vector{T}(g), h,
        collect(SolutionPhase, phases), collect(Int, idx_bounded), params,
        nq, gq, cq, hq, Matrix{T}(Aq),
        collect(Int, always_active), collect(Int, conservation_rows),
        Vector{T}(q0), Vector{T}(qscale),
    )
end

"""
    current_g(prob, q) -> Vector

The standard part of the gradient at the unknown parameters `q`: `prob.g` when
there are none, `prob.gq(q, prob.params)` otherwise.
"""
current_g(prob::DualNewtonProblem, q) =
    (prob.nq == 0 || prob.gq === nothing) ? prob.g : prob.gq(q, prob.params)

"""
    current_h(prob, x, q) -> Vector

The state-dependent part of the gradient. `prob.h(x, params)` unless the problem
declares `hq`, in which case `prob.hq(x, q, params)`.

An activity model can depend on the unknown parameters as well: the Debye-Hückel
coefficients are functions of temperature, so an adiabatic solve that left `h` at
the starting temperature would be minimizing the wrong Gibbs energy. Declaring it
explicitly is what keeps that dependence visible instead of routing it through a
mutated `params`, which the forward-mode Jacobian of the outer residual would not
differentiate.
"""
current_h(prob::DualNewtonProblem, x, q) =
    (prob.nq == 0 || prob.hq === nothing) ? prob.h(x, prob.params) :
    prob.hq(x, q, prob.params)

"""
    stationarity_capacity(prob) -> Int

The number of conservation rows `m`, which is the largest number of simultaneous
stationarity conditions the element potentials can carry.

This is Gibbs' phase rule, in the form the dual system takes. A bound-constrained
variable held ACTIVE contributes `uᵢ = gᵢ`, i.e. `aᵢᵀ y = −gᵢ`, one **linear**
equation in `y`; a mole-fraction mixing phase contributes
`logsumexp(uᵢ − gᵢ) = 0`, one more, nonlinear but still a condition on `y` alone.
A phase with a solvent does not, because its reference equation involves the
composition. With `y ∈ ℝᵐ` there is no `y` satisfying more than `m` of them, so an
active set carrying more cannot support a solution — the Newton residual cannot
reach zero for any iterate, and the least-squares step merely spreads the
violation over the rows.

That is not a slow case, it is an unsolvable one, and it has to be excluded by
construction rather than discovered. Measured on an LC³ equilibrium the active set
grew to 15 pure phases and 5 solid solutions — **19 conditions on 12 components** —
and the solve came back with a stationarity residual of 18 and an element balance
of 800 having never had a solution to find.
"""
stationarity_capacity(prob::DualNewtonProblem) = size(prob.A, 1)

"""
    _n_stationarity_conditions(prob, active, act_ph) -> Int

How many conditions on `y` alone the active set imposes: one per active
bound-constrained variable, one per active mole-fraction phase.
"""
function _n_stationarity_conditions(prob, active, act_ph)
    n = length(active)
    for k in act_ph
        prob.phases[k].mole_fraction && (n += 1)
    end
    return n
end

"""
    _active_set_supports_a_solution(prob, active, act_ph) -> Bool

Whether the stationarity block of this active set can be satisfied at some `y`.

Two conditions, both necessary. The count must not exceed
[`stationarity_capacity`](@ref). And the active bound-constrained variables'
composition vectors must be linearly INDEPENDENT: `A[:, active]ᵀ y = −g[active]`
is solvable for some `y` only then, since two dependent columns demand a fixed
relation between their `gᵢ` that the database will not happen to satisfy. That
second test is what catches two polymorphs of one composition — `Gbs` and
`AlOHmic` are both Al(OH)₃ — which impose `uᵢ = gᵢ` twice on the same vector.
"""
function _active_set_supports_a_solution(prob, active, act_ph)
    _n_stationarity_conditions(prob, active, act_ph) <= stationarity_capacity(prob) ||
        return false
    isempty(active) && return true
    # The rank is taken on the VALUE part, so that a `ForwardDiff.Dual` matrix — a
    # conservation matrix carrying a stoichiometric parameter, say the Mg/Al ratio
    # of a hydrotalcite or the Si substitution of a katoite — gives the same
    # answer as its primal. A rank IS a property of the value part: letting an
    # infinitesimal perturbation change which phases the algorithm considers
    # admissible would make the derivative of the answer discontinuous, which is
    # not a subtlety one wants to discover downstream.
    return LinearAlgebra.rank(ForwardDiff.value.(prob.A[:, active])) == length(active)
end

"""
    DualNewtonOptions(; tol, maxit, max_active_updates, si_tol, inner_tol,
                      inner_maxit, inner_fall_bound, lenient_line_search,
                      lp_fallback, verbose)

  - `tol`: tolerance on the KKT residual.
  - `maxit`: Newton iterations per active set.
  - `max_active_updates`: how many times the active set may change.
  - `si_tol`: saturation index above which a variable at its bound is admitted.
  - `inner_tol`, `inner_maxit`: the inner fixed point that recovers the phase
    compositions.
  - `inner_fall_bound`: the most a solute's log-amount may fall in one sweep of
    that fixed point, `30` by default, as a rise may. `Inf` lets a solute fall to
    what its potential asks at once.
  - `lenient_line_search`: when `true`, the first pass of the line search asks the
    candidate for a converged inner solve only where the current point has one.
  - `lp_fallback`: when no start converges, try once more from the vertex of the
    linear program (see [`dual_newton_solve`](@ref)). Off by default: a caller
    that runs its own search over starts, the linear program among them, would
    see a different start certify first.

The last two change the path of the search, not its answer's conditions, and
are off by default. They are for a caller that solves many neighboring problems
from warm starts, such as the steps of a kinetic run. There the inner fixed point
rarely converges at the points the Newton method passes through: a solute on its
way down moves by exactly the bound at every sweep, which the stall rule reads
as no progress, so the call stops before the solute arrives; and the first pass
of the line search then refuses forty candidates for want of what the current
point does not have either, before the second pass accepts the first of them.
Measured on the coupled hydration of a CEM I paste over three hours (82 steps),
the two together took the run from 176 s to 10 s on the same trajectory to the
last bit. They are not the default because they also change the path of a cold
solve, and a cold solve under the Debye-Hückel limiting law near its range is
path-sensitive: on one cement it went from certified to not.
"""
Base.@kwdef struct DualNewtonOptions
    tol::Float64 = 1.0e-10
    maxit::Int = 200
    max_active_updates::Int = 200
    si_tol::Float64 = 1.0e-8
    # The inner fixed point that recovers the phase compositions, and the
    # tolerance at which its answer may be trusted.
    #
    # `inner_tol` is what makes the outer residual a FUNCTION of `v`. The inner
    # loop is warm-started from the `W` it is handed, so its answer depends on
    # that warm start until it has converged; once it has, re-running it from its
    # own output changes nothing, and the residual depends on `v` alone. Steps
    # whose inner solve did not reach this are refused rather than accepted on a
    # residual that the next iteration will not reproduce.
    inner_tol::Float64 = 1.0e-10
    inner_maxit::Int = 200
    inner_fall_bound::Float64 = 30.0
    lenient_line_search::Bool = false
    lp_fallback::Bool = false
    verbose::Bool = false
end

# The same options with the fallback on the linear program turned off: the
# options of the solve that fallback runs, which must not fall back again.
_without_lp_fallback(o::DualNewtonOptions) = DualNewtonOptions(;
    (k => getfield(o, k) for k in fieldnames(DualNewtonOptions) if k !== :lp_fallback)..., lp_fallback = false,
)

# ── inner level: invert the stationarity of the log variables ─────────────────

# Composition of a mole-fraction phase from the potentials alone.
#
# `x_i = N · softmax(u_i − g_i − lnγ_i)` and the phase is consistent when
# `Σ_i exp(u_i − g_i − lnγ_i) = 1`, i.e. when the log-sum-exp vanishes. Both are
# written with the maximum factored out, which is the only form that survives the
# range of `u − g` a cement produces.
function _mole_fraction_exponents(prob, ph, u, hv, x_buf, N, dead, g = prob.g)
    d = Vector{promote_type(eltype(u), eltype(hv), eltype(x_buf), eltype(g), typeof(N))}(undef, length(ph.members))
    for (j, i) in enumerate(ph.members)
        if i in dead
            d[j] = -Inf
            continue
        end
        xi = x_buf[i]
        # `lnγ` is read at the current composition; identically zero for ideal
        # mixing, so the loop below converges in one pass there.
        lnγ = (xi > 0 && N > 0) ? hv[i] - log(xi / N) : zero(eltype(d))
        d[j] = u[i] - g[i] - lnγ
    end
    return d
end

# The saturation of a present mixing phase at the multipliers `u`, at the
# composition the inversion gave it: `uᵢ − ∇fᵢ` of its reference member, or the
# log-sum-exp of its exponents for a mole-fraction phase. Zero where its
# stationarity holds, negative where the phase is undersaturated.
function _phase_saturation(prob, k, u, hv, x_buf, g, dead)
    ph = prob.phases[k]
    if ph.mole_fraction
        N = sum(x_buf[i] for i in ph.members)
        return _logsumexp(_mole_fraction_exponents(prob, ph, u, hv, x_buf, N, dead, g))
    end
    ir = ph.members[ph.j_ref]
    return u[ir] - g[ir] - hv[ir]
end

# The value of a number with every level of duals removed: what a decision of an
# iteration (convergence, a stall, a step length) is taken on, so that a pass on
# dual numbers follows exactly the path of the pass on their values.
_primal_value(x) = x
_primal_value(x::ForwardDiff.Dual) = _primal_value(ForwardDiff.value(x))

function _logsumexp(d)
    M = maximum(d)
    isfinite(M) || return oftype(M, -Inf)
    return M + log(sum(exp(dj - M) for dj in d))
end

"""
    _newton_phase_composition(prob, ph, c, xt, f0, total, q, dead; maxit, tol)
        -> (f, L, converged)

The composition of one mole-fraction phase at the potentials `c = u − g` of its
members, by Newton's method: the mole fractions `f` and the constant `L` with

    h_j(f) − c_j + L = 0   for every live member j,     Σ_j f_j = 1,

`h` evaluated with the phase's members at `total · f` and every other variable
as in `xt`. These are the fixed-point equations of the substitution
`f = softmax(c − lnγ(f))`, and `L = logsumexp(c − lnγ)` is the tangent-plane
measure of the phase, so one routine serves the inversion and the stability
test. The unknowns are `z = ln f` and `L`; the Newton matrix is the bordered
`[H 1; fᵀ 0]` with `H = ∂h/∂z` from `ForwardDiff`, and each step is accepted on
a decrease of the squared residual, halving it otherwise.

A member at the floor whose residual is positive asks for less than the floor
allows. It is held there and left out of the system, which is what the
substitution does too: its fraction is `exp(W_FLOOR)`, zero to the precision
of everything it enters.

A bounded member (`bounded_members`) whose residual is positive, and would
still be by the slope of its activity at the floor, is set there, and held if
it still asks for less: that is its condition of absence, the inequality. Its
activity stays finite as it vanishes, so its own derivative is nearly zero there
and the Newton step cannot carry it down.

`f0` is the start, over `ph.members`; dead members get a zero fraction.
"""
function _newton_phase_composition(
        prob, ph, c, xt, f0, total, q, dead; maxit::Int = 50, tol::Float64 = 1.0e-13,
    )
    nm = length(ph.members)
    live = [j for j in 1:nm if !(ph.members[j] in dead)]
    # In the number type of the potentials and of the composition: the outer
    # Jacobian is taken through this iteration when the inversion around it has
    # not converged (`_swept_jacobian`).
    T = promote_type(eltype(c), eltype(xt), typeof(total))
    f = zeros(T, nm)
    K = length(live)
    K == 0 && return (f, -Inf, true)
    mem = ph.members[live]
    cl = [c[j] for j in live]
    xw = Vector{T}(xt)
    for (j, i) in enumerate(ph.members)
        j in live || (xw[i] = 0.0)
    end
    lh = ph.local_h
    function hfun(zz)
        if lh === nothing
            xd = Vector{eltype(zz)}(xw)
            for (a, i) in enumerate(mem)
                xd[i] = total * exp(zz[a])
            end
            return current_h(prob, xd, q)[mem]
        end
        xm = zeros(eltype(zz), nm)
        for (a, j) in enumerate(live)
            xm[j] = total * exp(zz[a])
        end
        return lh(xm)[live]
    end
    z = [log(max(f0[j], exp(W_FLOOR))) for j in live]
    z .-= _logsumexp(z)
    held(zz, r) = [zz[a] <= W_FLOOR + 1 && r[a] > 0 for a in 1:K]
    function state(zz)
        hz = hfun(zz)
        L = _logsumexp(cl .- hz .+ zz)
        r = hz .- cl .+ L
        out = held(zz, r)
        φ = sum(abs2(r[a]) for a in 1:K if !out[a]; init = zero(eltype(r)))
        return (; hz, L, r, out, φ)
    end
    st = state(z)
    # The bounded members among the live ones, by position in `z`.
    bnd = [a for (a, j) in enumerate(live) if j in ph.bounded_members]
    converged = false
    for _ in 1:maxit
        # Without a closure over `st`, which the iteration reassigns and Julia
        # would then box.
        free = findall(!, st.out)
        if isempty(free) || maximum(abs, view(st.r, free)) <= tol
            converged = true
            break
        end
        H = ForwardDiff.jacobian(hfun, z)
        # A bounded member asking for less that would still ask for less at the
        # floor, by the slope its activity has here, is absent: it is set there
        # and held if the exact residual agrees. Left to the Newton step, whose
        # length its nearly flat activity limits, a member of the gel of Myers
        # et al. barely unstable in a C-A-S-H stopped at 1e-17 to 1e-28 mol,
        # between the floor of this iteration and that of the certificate, which
        # then judged it by the equality of a present member and refused it. A
        # member present at a fraction that counts has a slope that makes the
        # test fail, and is left to the step.
        snapped = false
        for a in bnd
            (st.out[a] || st.r[a] <= 0) && continue
            _primal_value(st.r[a]) - _primal_value(H[a, a]) * (_primal_value(z[a]) - W_FLOOR) > 0 || continue
            zt = copy(z)
            zt[a] = W_FLOOR
            zt .-= _logsumexp(zt)
            stt = state(zt)
            if stt.r[a] > 0
                z, st, snapped = zt, stt, true
            end
        end
        snapped && continue
        nf = length(free)
        B = zeros(eltype(H), nf + 1, nf + 1)
        B[1:nf, 1:nf] .= @view H[free, free]
        B[1:nf, nf + 1] .= 1.0
        for (p, a) in enumerate(free)
            B[nf + 1, p] = exp(z[a])
        end
        rhs = vcat(-st.r[free], 1.0 - sum(exp, z))
        sol = qr(B, ColumnNorm()) \ rhs
        all(isfinite, sol) || break
        δ = sol[1:nf]
        # The step length is a decision of the iteration, taken on the values.
        α = min(1.0, 30.0 / max(maximum(a -> abs(_primal_value(a)), δ), eps()))
        accepted = false
        for _ in 1:30
            zt = copy(z)
            for (p, a) in enumerate(free)
                zt[a] = clamp(z[a] + α * δ[p], W_FLOOR, 0.0)
            end
            zt .-= _logsumexp(zt)
            stt = state(zt)
            if isfinite(stt.φ) && stt.φ <= (1 - 1.0e-4 * α) * st.φ
                z, st, accepted = zt, stt, true
                break
            end
            α /= 2
        end
        accepted || break
    end
    for (a, j) in enumerate(live)
        f[j] = exp(z[a])
    end
    return (f, st.L, converged)
end

function _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)
    fill!(x_buf, 0.0)
    for (a, k) in enumerate(act_ph)
        ph = prob.phases[k]
        for (j, i) in enumerate(ph.members)
            x_buf[i] = (!ph.mole_fraction && j == ph.j_ref) ? refs[a] : exp(W[k][j])
        end
    end
    @inbounds for (j, i) in enumerate(active)
        x_buf[i] = xB[j]
    end
    return x_buf
end

# The variables held at the floor because a component they carry is absent from
# the budget (`_degenerate_conservation_rows`), as a mask: the sweeps ask about
# every member at every sweep, and a mask answers without hashing.
struct _DeadSet <: AbstractSet{Int}
    mask::BitVector
    count::Int
end
Base.in(i::Integer, d::_DeadSet) = d.mask[i]
Base.length(d::_DeadSet) = d.count
function Base.iterate(d::_DeadSet, from::Int = 1)
    i = findnext(d.mask, from)
    return i === nothing ? nothing : (i, i + 1)
end

function _dead_variables(A::AbstractMatrix, degenerate)
    mask = falses(size(A, 2))
    for j in axes(A, 2)
        mask[j] = any(abs(A[k, j]) > 0 for k in degenerate)
    end
    return _DeadSet(mask, count(mask))
end

"""
    _invert_phases!(prob, W, y, refs, act_ph, active, xB, x_buf; dead) -> x_buf

Recover the members of every ACTIVE mixing phase from their own stationarity,
`hᵢ(x) = uᵢ − gᵢ`, at fixed multipliers and fixed phase references.

For a variable whose `hᵢ` behaves like `ln xᵢ` plus a slowly varying part — which
is what a logarithm of a mole fraction is — `∂hᵢ/∂wᵢ = 1`, so `w += r` is an
exact Newton step; the coupling through the phase total and the activity
coefficients is what makes the loop iterative rather than one-shot.

The floor is `exp(-700)` (`W_FLOOR`) and not higher: a variable whose
stationarity demands `hᵢ = −200` could never reach a floor of `−80`, its residual
would stay at 120 for ever, and the loop could never report convergence — which
would then make the OUTER Jacobian meaningless.
"""
function _invert_phases!(
        prob, W, y, refs, act_ph, active, xB, x_buf;
        dead = Set{Int}(), g = prob.g, q = prob.q0,
        resid::Union{Nothing, Base.RefValue{Float64}} = nothing, maxsweeps::Int = 200,
        max_fall::Float64 = 30.0, trial::Bool = false,
        composed::Union{Nothing, AbstractVector{Bool}} = nothing,
    )
    u = -(transpose(prob.A) * y)
    worst = Inf
    smallest = Inf
    stalled = 0
    # A solute pushed to the ceiling: these potentials hold no composition (see
    # below, after the sweep).
    runaway = false
    # The phases recovered by their own `invert`, once: its inputs, the
    # potentials and the amount of the reference, do not change from one sweep
    # to the next.
    inverted = falses(length(prob.phases))
    # Whether each phase's own inversion is used. A point that is not a trial of
    # the line search (the iterate itself, a start) needs a composition to move
    # from whether or not its potentials hold one: where the inversion finds
    # none there, the sweeps take over, as they did before it existed, and
    # `composed[k]` records which of the two it was. A trial stops at once
    # instead, which is where nearly all the inversions are, but only where the
    # iterate had a composition: the trial has then left the potentials that
    # hold one, and a shorter step comes back to them. From an iterate whose
    # potentials hold none, every nearby trial holds none either, and stopping
    # each of them stopped the solve at its first iterate: a paste loaded with
    # sodium chloride under the limiting law, which the sweeps had carried from
    # that same start to its answer. Its trials are swept, as the iterate was.
    use_invert = [prob.phases[k].invert !== nothing for k in eachindex(prob.phases)]
    # The phases taken by their own inversion in the sweep under way.
    by_inversion = falses(length(prob.phases))
    # `h` is needed by the phases that are swept, and only by them.
    swept = any(k -> !use_invert[k], act_ph)

    for _ in 1:maxsweeps
        _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)
        worst = 0.0
        # Each phase reads the composition the sweep starts from and writes its
        # own, so the phases can be taken in any order. Those that invert
        # themselves come first: they need no `h`, and a trial whose inversion
        # runs away ends there, before the activities are evaluated and the
        # other phases swept, all of which it would throw away.
        hv_needed = swept
        fill!(by_inversion, false)
        for (a, k) in enumerate(act_ph)
            trial && runaway && break
            ph = prob.phases[k]
            (!ph.mole_fraction && use_invert[k]) || continue
            by_inversion[k] = true
            # The solutes in one call, exactly, by the phase's own inversion
            # (see `SolutionPhase`); `nothing` is a model with no solution at
            # these potentials, which the sweep would only cycle on.
            inverted[k] && continue
            inverted[k] = true
            # A member whose component is absent from the budget (`dead`) is
            # held at the floor by the sweep, and is no amount here: `-Inf`.
            # Counted with the potential that pins its row, it would carry the
            # ionic strength away on its own: a pore solution without a redox
            # carrier came back as given, every start judged to hold nothing.
            c = [i in dead ? oftype(u[i] - g[i], -Inf) : u[i] - g[i] for i in ph.members]
            wn = ph.invert(c, refs[a], W[k], q, prob.params)
            trial || composed === nothing || (composed[k] = wn !== nothing)
            if wn === nothing
                if trial && (composed === nothing || composed[k])
                    runaway = true
                else
                    use_invert[k] = false
                    swept = true
                    worst = max(worst, one(worst))   # the sweeps start next
                end
                continue
            end
            for (j, i) in enumerate(ph.members)
                (j == ph.j_ref || i in dead) && continue
                w = clamp(wn[j], W_FLOOR, W_CEIL)
                worst = max(worst, abs(_primal_value(w - W[k][j])))
                W[k][j] = w
                trial && w >= W_CEIL && (runaway = true)
            end
        end
        if !(trial && runaway)
            hv = hv_needed ? current_h(prob, x_buf, q) : nothing
            for (a, k) in enumerate(act_ph)
                by_inversion[k] && continue
                ph = prob.phases[k]
                if ph.mole_fraction && ph.newton
                    # The same composition as the substitution below, found by
                    # Newton's method: see `SolutionPhase`. Solved to convergence
                    # at each sweep, from the previous one, so a second sweep finds
                    # it already there.
                    N = refs[a]
                    c = [u[i] - g[i] for i in ph.members]
                    f0 = [exp(W[k][j]) / N for j in eachindex(ph.members)]
                    f, _, _ = _newton_phase_composition(prob, ph, c, x_buf, f0, N, q, dead)
                    for j in eachindex(ph.members)
                        w = f[j] > 0 ? clamp(log(N) + log(f[j]), W_FLOOR, 700.0) : W_FLOOR
                        worst = max(worst, abs(_primal_value(w - W[k][j])))
                        W[k][j] = w
                    end
                elseif ph.mole_fraction
                    # No member of a solid solution can be inverted from its own
                    # stationarity: `hᵢ` is the logarithm of a mole fraction, bounded
                    # above by zero, so a positive `uᵢ − gᵢ` is unreachable and the
                    # iteration answers by growing the member without bound. What the
                    # potentials DO determine, always, is the composition: the
                    # fractions are the softmax of `u − g − lnγ`, and the phase's
                    # total is the outer unknown. Nothing here can overflow.
                    # `Nf`, not `N`: the branch above binds `N` too, its
                    # comprehension captures it, and a captured binding assigned
                    # twice is boxed.
                    Nf = refs[a]
                    d = _mole_fraction_exponents(prob, ph, u, hv, x_buf, Nf, dead, g)
                    M = maximum(_primal_value, d)
                    isfinite(M) || continue          # every member of the phase is dead
                    lZ = M + log(sum(exp(dj - M) for dj in d))
                    for (j, _) in enumerate(ph.members)
                        w = clamp(log(Nf) + d[j] - lZ, W_FLOOR, 700.0)
                        worst = max(worst, abs(_primal_value(w - W[k][j])))
                        W[k][j] = w
                    end
                else
                    # An aqueous solution has a solvent, and the solutes carry
                    # molalities that are unbounded above, so each of them IS
                    # recoverable from its own stationarity — and must be, because
                    # that is where their dependence on the potentials lives.
                    for (j, i) in enumerate(ph.members)
                        (j == ph.j_ref || i in dead) && continue
                        r = (u[i] - g[i]) - hv[i]
                        w = clamp(W[k][j] + clamp(r, -max_fall, W_MAX_RISE), W_FLOOR, W_CEIL)
                        # THE MEASURE IS THE STEP, NOT THE RESIDUAL, and that is what
                        # `inner_tol` actually asks for: "re-running this loop from
                        # its own output changes nothing". That is a statement about
                        # how far the STATE moves, and the residual is not it.
                        #
                        # The two differ exactly where it matters. `W` is a
                        # log-molality clamped to `[W_FLOOR, W_CEIL]`, and a species
                        # whose stationarity asks for less than the floor can never
                        # reach it: the update is clamped, the state does not move,
                        # and the residual stays where it is FOR EVER. Measured on a
                        # CEM I at its own converged answer, the worst residual was
                        # 1.0054e+02 at sweep 1 and 1.0054e+02 at sweep 200 -- it
                        # belonged to dissolved oxygen at 9.9e-305 mol, which a
                        # reducing pore solution puts well below the floor. Every one
                        # of the 309 calls in a warm solve burned all 200 sweeps on
                        # it, and that was 98 % of the solve.
                        #
                        # Measuring the step needs no special case for those species:
                        # a clamped update moves nothing, so it contributes exactly
                        # zero. Nor does it need them EXCLUDED, which would be wrong
                        # -- a species sits at the floor only while the others hold
                        # it there, and excluding it lets the loop stop one sweep
                        # before it comes off.
                        worst = max(worst, abs(_primal_value(w - W[k][j])))
                        W[k][j] = w
                        trial && w >= W_CEIL && (runaway = true)
                    end
                end
            end
        end

        # A solute at the ceiling, `e^20` mol, is an inversion running away and
        # not one converging: the potentials ask for more of it than any budget
        # holds. Measured on cement pastes under the limiting law past its range,
        # 91 % of 200 000 inversions ended unconverged, cycling with a period of
        # about eight sweeps between the ceiling and a fall of thirty while the
        # ionic strength swung with it, each until the stall rule stopped it twenty
        # sweeps later. No sweep after the first touch changes the verdict, so a
        # trial of the line search stops there and reports an infinite step.
        if runaway
            worst = Inf
            break
        end
        # Every phase recovered by its own inversion is exact after one call:
        # sweeping again would change nothing.
        swept || (worst = 0.0)
        worst <= 1.0e-14 && break
        # Not converging: see `INNER_STALL_SWEEPS`.
        if worst < smallest
            smallest = worst
            stalled = 0
        else
            stalled += 1
            stalled >= INNER_STALL_SWEEPS && break
        end
    end

    # What the sweeps ended on, so the caller can tell a converged inversion from
    # one that merely ran out of sweeps. Left unreported, the two are
    # indistinguishable and the second silently makes the outer residual depend
    # on the warm start rather than on `v`.
    resid === nothing || (resid[] = worst)

    return _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)
end

"""
    _outer_residual(prob, v, W, act_ph, active, b, x_buf; dead, degenerate) -> R

The outer residual at `v = [ln x_ref(φ) for each active phase; y; x_B; q]`:

  - stationarity of each active phase's reference — one equation per phase, which
    is what fixes that phase's total amount;
  - the equality constraints `A x − b` — `m` equations;
  - stationarity of the active bound-constrained variables — `|P|` equations;
  - `prob.cq(x, q, params)` — one equation per unknown parameter.

`|Φ| + m + |P| + nq` equations in as many unknowns: some fifteen for a cement,
against forty-seven variables in the interior-point route.

The `q` block is how a prescribed property is imposed: an adiabatic solve has `T`
in `q` and `H(x, T) − H₀` in `cq`; a fixed-volume solve has `P` in `q` and
`V(x, P) − V₀`; a kinetic step has the reaction extents in `q` and
`Δξ − Δt·M·r(x)`. They are unknowns of the SAME system as the amounts and the
multipliers, not an outer loop around it, which is the structure Reaktoro uses
and the reason a kinetic step costs `nq` extra equations rather than a second
solve.
"""
function _outer_residual(
        prob, v, W, act_ph, active, b, x_buf;
        dead = Set{Int}(), degenerate = Int[],
        resid::Union{Nothing, Base.RefValue{Float64}} = nothing,
        inner_maxit::Int = 200, max_fall::Float64 = 30.0, trial::Bool = false,
        composed::Union{Nothing, AbstractVector{Bool}} = nothing,
    )
    m = size(prob.A, 1)
    nph = length(act_ph)
    na = length(active)
    nq = prob.nq

    refs = [exp(v[a]) for a in 1:nph]
    y = v[(nph + 1):(nph + m)]
    xB = v[(nph + m + 1):(nph + m + na)]
    q = v[(nph + m + na + 1):(nph + m + na + nq)]
    g = current_g(prob, q)

    _invert_phases!(
        prob, W, y, refs, act_ph, active, xB, x_buf; dead = dead, g = g, q = q,
        resid = resid, maxsweeps = inner_maxit, max_fall, trial, composed,
    )
    # A trial whose inversion ran away holds no composition, and the line search
    # reads no more than that of it.
    trial && resid !== nothing && isinf(resid[]) &&
        return promote_type(eltype(v), eltype(x_buf), eltype(g))[]

    hv = current_h(prob, x_buf, q)
    u = -(transpose(prob.A) * y)

    # One equation per active phase, fixing its absolute level.
    #
    # With a solvent that is the reference member's own stationarity. Without one
    # it is the statement that the mole fractions sum to unity, written as a
    # log-sum-exp — which is the same quantity the tangent-plane test uses to
    # decide whether the phase may form at all, so admission and stationarity are
    # measured by one expression rather than two that could disagree.
    R_ref = promote_type(eltype(v), eltype(x_buf), eltype(hv), eltype(g))[]
    for (a, k) in enumerate(act_ph)
        ph = prob.phases[k]
        if ph.mole_fraction
            d = _mole_fraction_exponents(prob, ph, u, hv, x_buf, refs[a], dead, g)
            push!(R_ref, _logsumexp(d))
        else
            ir = ph.members[ph.j_ref]
            push!(R_ref, g[ir] + hv[ir] - u[ir])
        end
    end

    # `A x + Aq q − b`: a linear relation may involve the unknown parameters as
    # well. That is how a reaction extent enters — the reactivity constraint
    # `Kᵀn − Δξ = ξ₀` is LINEAR, so it belongs in the conservation block beside
    # the elements and the charge, and the algebraic cost of kinetics is the
    # number of reactions rather than the number of species.
    Rb = nq == 0 ? (prob.A * x_buf .- b) : (prob.A * x_buf .+ prob.Aq * q .- b)

    # The stationarity of an active bound-constrained variable is `gᵢ + hᵢ = uᵢ`,
    # not `gᵢ = uᵢ`. The two coincide only when `hᵢ = 0`, the case of a pure phase;
    # writing the general form costs nothing there and is the only correct one
    # otherwise.
    Rs = g[active] .+ hv[active] .- u[active]

    # For a vanished component the balance row carries no information — every
    # variable containing it is pinned at the floor, so the row reads 0 = 0 and
    # leaves `y_k` free, making the Jacobian singular. Replace it by the equation
    # that fixes the multiplier, which keeps the system square.
    for k in degenerate
        Rb[k] = y[k] - DEGENERATE_POTENTIAL
    end

    Rq = nq == 0 ? eltype(R_ref)[] : prob.cq(x_buf, q, prob.params)
    length(Rq) == nq || throw(
        DimensionMismatch(
            "`cq` returned $(length(Rq)) residuals for $nq unknown " *
                "parameters; the system would not be square."
        )
    )

    return vcat(R_ref, Rb, Rs, Rq)
end

# The admission measure of an absent mixing phase: Michelsen's, the one the
# certificate applies (`kkt_certificate`).
#
# The search admits a phase on the measure the certificate applies, not on the
# ideal sum `Σᵢ exp(uᵢ − gᵢ) − 1` with `lnγ = 0` and every member counted. The two
# disagree where it matters. For a non-ideal phase (a Redlich-Kister binary) the
# ideal test and the certificate can reach opposite verdicts on the same phase,
# so the search would hold absent what the certificate then calls supersaturated,
# or the reverse. And a DEAD member — one whose component is absent from the
# budget, its potential pinned at the sentinel `DEGENERATE_POTENTIAL` — would make
# any phase holding it look supersaturated whatever the chemistry: the `CSHQ` of a
# binder without potassium, for one. `phase_tangent_trial` excludes dead members.
_admission_measure(prob, k, u, x_buf, q, dead) =
    phase_tangent_measure(prob, k, u, x_buf; g = current_g(prob, q), q = q, dead = dead)

# The same, kept in `memo[k]` for the rest of an active-set round, within which
# the multipliers, the composition and the parameters it is measured at do not
# change.
function _admission_measure!(memo, prob, k, u, x_buf, q, dead)
    isnan(memo[k]) && (memo[k] = _admission_measure(prob, k, u, x_buf, q, dead))
    return memo[k]
end

"""
    phase_tangent_trial(prob, k, u, x; maxit = 50, tol = nothing, total = 1e-6,
                        g = prob.g, q = prob.q0, start = nothing, dead = Set{Int}())
        -> (measure, fractions)

Michelsen's tangent-plane measure for mixing phase `k` at the multipliers `u` and
composition `x`, with the trial composition that attains it: the log-sum-exp of
`uᵢ − gᵢ − lnγᵢ` over the phase's members, the trial composition refined against
the phase's OWN activity model — by successive substitution, or by Newton's method
for a phase declared `newton = true` ([`SolutionPhase`](@ref)).

`fractions` are mole fractions over the phase's members. `g` and `q` are the
standard part of the gradient and the unknown parameters `h` is evaluated at;
`start` the trial composition to begin from (uniform over the live members by
default); `dead` the members whose component is absent from the budget, which
take no part. `tol` is the stopping tolerance on the fractions, by default that
of each iteration: `1e-12` for the substitution, `1e-13` for Newton's method.

A verdict, taken on values: dual numbers carried by `u`, `x` or what `h`
returns are dropped, and both results are `Float64`.

Positive means a trial composition of the phase lies below the tangent plane, so
the phase can form and a composition without it is not optimal. Zero means the
phase is exactly at its stability limit.

This is what an absent **mixing** phase requires, and it is not the sign of a
saturation index: a solution phase has no single index, and its members are never
exactly zero while it exists — which is why `kkt_certificate` cannot test them
one by one and needs this instead.

`total` is the trial phase amount. The measure is independent of it for an ideal
phase, and for a non-ideal one it sets the composition at which `γ` is read; a
small value is the right choice, since the question is whether an *infinitesimal*
amount of the phase is stable.
"""
function phase_tangent_trial(
        prob::DualNewtonProblem, k::Int, u::AbstractVector, x::AbstractVector;
        maxit::Int = 50, tol::Union{Nothing, Float64} = nothing, total::Float64 = 1.0e-6,
        g = prob.g, q = prob.q0, start = nothing, dead = Set{Int}(),
    )
    ph = prob.phases[k]
    nm = length(ph.members)
    nm == 0 && return (-Inf, Float64[])
    # A DEAD member is one whose conservation row is degenerate -- no matter of
    # that component exists in the system, so the row's multiplier is pinned at
    # the sentinel `DEGENERATE_POTENTIAL` rather than solved for. `uᵢ` for such a
    # member is therefore not a chemical potential at all, and `uᵢ - gᵢ` is a
    # number with no meaning. Feeding it to the measure below reports a phase as
    # wanting to move to a composition THE ELEMENT BALANCE FORBIDS.
    #
    # Measured: the CEM III/A of the documentation declares `CSHQ` with its
    # alkali end-members `KSiOH` and `NaSiOH`, and its slag and clinker bring no
    # potassium and no sodium. Without this guard the measure returned +54.06 on
    # a converged, mass-balanced equilibrium -- the same +54.06 on every other
    # system with the same declaration, which is the signature of a sentinel
    # rather than of chemistry. `_mole_fraction_exponents` had this guard from
    # the start; this function was written without it.
    live = [j for j in 1:nm if !(ph.members[j] in dead)]
    length(live) < 1 && return (-Inf, Float64[])
    # A verdict, taken on values like every decision of the search: the
    # composition, the multipliers and `h` may carry dual numbers (the
    # certificate of a problem whose `h` captures the data being
    # differentiated), and only their values are measured.
    xt = Float64[_primal_value(v) for v in x]
    # The successive substitution below converges to the stationary point of the
    # tangent-plane distance NEAREST its start, so which start is used decides
    # which one is found. A uniform start is the natural probe for a phase held
    # absent. It is not enough for a phase that is PRESENT: there the phase's own
    # composition is a stationary point with measure zero, and the uniform start
    # runs straight to it, reporting nothing however unstable the phase is. Such
    # a phase needs starts elsewhere — see `phase_split_measure`.
    frac = if start === nothing
        f = zeros(nm)
        for j in live
            f[j] = 1.0 / length(live)
        end
        f
    else
        f = collect(start)
        for j in 1:nm
            j in live || (f[j] = 0.0)
        end
        sf = sum(f)
        sf > 0 ? f ./ sf : f
    end
    if ph.newton
        c = Float64[_primal_value(u[i] - g[i]) for i in ph.members]
        f, L, _ = tol === nothing ?
            _newton_phase_composition(prob, ph, c, xt, frac, total, q, dead; maxit = maxit) :
            _newton_phase_composition(prob, ph, c, xt, frac, total, q, dead; maxit = maxit, tol = tol)
        return (_primal_value(L), Float64[_primal_value(v) for v in f])
    end
    lnZ = -Inf
    d = Vector{Float64}(undef, nm)
    for _ in 1:maxit
        for (j, i) in enumerate(ph.members)
            xt[i] = total * frac[j]
        end
        hv = current_h(prob, xt, q)
        for (j, i) in enumerate(ph.members)
            if !(j in live)
                d[j] = -Inf
                continue
            end
            # `hᵢ` is `ln aᵢ = ln(xᵢ/N) + lnγᵢ`, so `lnγ` is what remains once the
            # ideal part is removed.
            lnγ = xt[i] > 0 ? _primal_value(hv[i]) - log(xt[i] / total) : 0.0
            d[j] = _primal_value(u[i] - g[i]) - lnγ
        end
        M = maximum(d)
        isfinite(M) || return (-Inf, Float64[])
        # The comprehension reads a binding of the iteration, not `lnZ`, which
        # it reassigns and Julia would then box.
        lz = M + log(sum(exp(dj - M) for dj in d))
        lnZ = lz
        newfrac = [exp(dj - lz) for dj in d]
        Δ = maximum(abs, newfrac .- frac)
        frac = newfrac
        Δ < something(tol, 1.0e-12) && break
    end
    return (lnZ, frac)
end

"""
    phase_tangent_measure(prob, k, u, x; kwargs...) -> Float64

The tangent-plane distance alone, for a caller that does not need the trial
composition. Identical to the first element of [`phase_tangent_trial`](@ref).
"""
phase_tangent_measure(args...; kwargs...) = first(phase_tangent_trial(args...; kwargs...))

"""
    phase_split_trial(prob, k, u, x; kwargs...) -> (Float64, Vector{Float64})

How far a **present** mixing phase is from wanting to split into two coexisting
compositions — and **the composition it wants to split into**.

The first element is the largest tangent-plane distance found from any start
other than the phase's own composition. The second is the trial composition that
achieved it, as mole fractions over the phase's members: Michelsen's stability
analysis does not only answer *whether* a phase splits, it hands the flash that
follows its starting estimate, and discarding it wastes the expensive half of the
calculation. A caller that means to act on the verdict — by seeding a second
instance of the phase — needs exactly this vector.

A positive value means a composition exists that lies **below** the tangent plane
at the current one, so the Gibbs minimum for that phase is not one composition
but two, and an answer reporting a single one is not the minimum however well it
satisfies the stationarity conditions of its members.

This is the case a **miscibility gap** produces, and it is not hypothetical for a
cement: the published Redlich-Kister parameters of the AFm sulfate/hydroxide and
AFt sulfate/carbonate binaries are concave over an interval, so a phase sitting
in that interval unmixes.

Why it needs its own function rather than a call to
[`phase_tangent_measure`](@ref) with the default start: at equilibrium the
members of a present phase satisfy `uᵢ = gᵢ + hᵢ`, so the tangent-plane distance
at the phase's own composition is exactly zero and is a stationary point of the
iteration. Started uniformly, the search converges to it and reports zero. The
starts used here are the **end-member corners**, which lie in the other lobe
when there is one.

Returns `(-Inf, Float64[])` for a phase of fewer than two members, which cannot
split.

# Keywords

`maxit`, `tol`, `total`, `g`, `q` and `dead` are those of
[`phase_tangent_trial`](@ref), run from every start. `corner` is the fraction of
the dominant member at each corner start (`0.98`: a corner of the simplex is
itself a fixed point of the substitution). `starts` adds trial compositions to the
corners and to the phase's own `split_starts`, as mole fractions over its members;
dual numbers among them are taken by value.
"""
function phase_split_trial(
        prob::DualNewtonProblem, k::Int, u::AbstractVector, x::AbstractVector;
        maxit::Int = 50, tol::Union{Nothing, Float64} = nothing, total::Float64 = 1.0e-6,
        g = prob.g, q = prob.q0, corner::Float64 = 0.98, dead = Set{Int}(),
        starts = (),
    )
    ph = prob.phases[k]
    nm = length(ph.members)
    nm < 2 && return (-Inf, Float64[])
    # Only the members the element balance can actually supply. A corner on a
    # dead member is not a composition the phase can take, so it is not evidence
    # that the phase wants to split -- and its `uᵢ` is a sentinel besides. With
    # fewer than two live members there is no interior to unmix into.
    live = [j for j in 1:nm if !(ph.members[j] in dead)]
    length(live) < 2 && return (-Inf, Float64[])
    worst = -Inf
    best = Float64[]

    # The corners of the simplex, and whatever the caller supplies.
    #
    # THE CORNERS ARE NOT ALWAYS ENOUGH. The successive substitution converges to
    # the stationary point of the tangent-plane distance NEAREST its start, so
    # the start decides which one is found, and a corner is not always on the
    # right side of the barrier. Where it is not, the iteration walks back to the
    # phase's own composition and the function reports nothing, though a split
    # would lower the energy.
    #
    # Measured, on the AFm sulfate/hydroxide binary with the published
    # Redlich-Kister parameters (A₀ = 0.188, A₁ = 2.49 in RT units; spinodal
    # [0.631, 0.914], binodal [0.4999, 0.9700]):
    #
    #     phase at x = 0.5268   corners +3.99e-02 (trial x = 0.974)   FOUND
    #     phase at x = 0.95     corners +2.43e-16 (trial x = 0.950)   MISSED,
    #                           binodal start +1.23e-01 (trial x = 0.444)
    #
    # Both compositions are metastable — inside the binodal, outside the
    # spinodal — so being metastable is not by itself what defeats the corners;
    # sitting in the lobe the corners lead back into is. That is not something
    # this function can know in advance, which is why the escape hatch exists
    # rather than a rule for when to use it.
    #
    # `starts` is how a caller that knows the mixing model hands over a better
    # one. For a binary the binodal of the ISOLATED model — a common-tangent
    # construction, microseconds — lands in the other lobe by construction, and
    # the search here then refines it with the chemical potentials the full
    # system actually has. The division of labor is deliberate: this function
    # knows the system and not the model, the caller knows the model and not the
    # system. Extra starts can only raise the maximum this function returns, so
    # supplying them is never a change of verdict on a phase the corners already
    # flagged — only a chance at one they did not.
    trials = Vector{Vector{Float64}}()
    for j in live
        # Nearly pure in member `j`, the rest shared among the live ones. Not
        # exactly pure: a corner of the simplex is itself a fixed point of the
        # substitution.
        frac = zeros(nm)
        for l in live
            frac[l] = (1 - corner) / (length(live) - 1)
        end
        frac[j] = corner
        push!(trials, frac)
    end
    for st in Iterators.flatten((ph.split_starts, starts))
        length(st) == nm || continue
        push!(trials, Float64[_primal_value(v) for v in st])
    end

    for frac in trials
        m, f = phase_tangent_trial(
            prob, k, u, x; maxit = maxit, tol = tol, total = total,
            g = g, q = q, start = frac, dead = dead,
        )
        if m > worst
            worst = m
            best = f
        end
    end
    return (worst, best)
end

"""
    phase_split_measure(prob, k, u, x; kwargs...) -> Float64

The split measure alone, for a caller that does not need the incipient
composition. Identical to the first element of [`phase_split_trial`](@ref).
"""
phase_split_measure(args...; kwargs...) = first(phase_split_trial(args...; kwargs...))

# ── the solve ─────────────────────────────────────────────────────────────────

"""
    simplex_start(A, g, b; floor = 0.0, maxit = 0) -> Union{Nothing, Vector}

A feasible composition to start from: the vertex of the mass balance that the
linear program `minimize gᵀx subject to A x = b, x ≥ 0` lands on, as
[`lp_start`](@ref) computes and verifies it. Returns that vertex, raised to
`floor`, when the program is solved; `nothing` when it is **proved** infeasible,
which is a statement about `b` rather than about the solver; and throws when the
tableau's answer could not be verified either way, which the earlier
implementation reported as one or the other without checking.

# What the vertex is worth as a start for the dual Newton

Measured, because the transplant was tried before it was believed. On a
91-species Portland cement with 12 components (OptimaSolver 0.6.0):

| | |
|:--|--:|
| the LP itself | 0.27 s, `‖A x − b‖∞ = 6e-17`, 6 nonzeros of 91 |
| cold start, the solver's own cascade | 16.1 s, certified |
| the same cascade started from this vertex | 14.4 s, certified |
| this vertex with the cascade declined | 10.5 s, **not** certified |

A vertex has at most `m` nonzero species, so every other one sits at the floor,
the configuration a log-domain method handles worst. The multipliers and the
basis of the same program are a different start; [`lp_start`](@ref) returns
them.
"""
function simplex_start(
        A::AbstractMatrix, g::AbstractVector, b::AbstractVector;
        floor::Float64 = 0.0, maxit::Int = 0,
    )
    lp = lp_start(A, g, b; maxit = maxit)
    lp.status === :optimal && return max.(lp.x, floor)
    lp.status === :infeasible && return nothing
    throw(
        ErrorException(
            "simplex_start: the linear program could not be decided ($(lp.status)); " *
                "its tableau reached an answer that does not verify on the data.",
        ),
    )
end

"""
    _element_potential_start(A, g, b, y0; dead, maxit, tol) -> y

Element potentials from the CONCAVE dual of the ideal problem — Brinkley's method,
after White, Johnson & Dantzig (1958).

Treat every species as an ideal one whose activity is its own amount. The
Lagrangian then minimizes in closed form, `xᵢ = exp(uᵢ − gᵢ)` with `u = −Aᵀy`, and
the dual becomes

```
φ(y) = −bᵀ y − Σᵢ exp(uᵢ − gᵢ),
∇φ  = A x − b,
∇²φ = −A diag(x) Aᵀ ≺ 0 .
```

`φ` is smooth and **strictly concave**, so Newton with a backtracking line search
converges from any starting point — there is no active set, no combinatorics, and
no way to stall in a wrong one. Its solution is the `y` for which the ideal
composition conserves matter exactly.

That is not the model this solver goes on to solve — the real phases carry mole
fractions and molalities, not bare amounts — but it is the right place to start
from, and it is what the three least-squares fits were standing in for. Those fits
minimize `‖Aᵀy + g + h‖` over a subset of species, which says nothing about the
element budget; the potentials they produce are consistent with no composition in
particular. Measured on an LC³ equilibrium at a quarter of full reaction, the inner
Newton failed to converge on EVERY active set the search visited, with residuals
between 8 and 19, and the same problem reached by continuation from a nearby
solution converged to 1e-11.
"""
function _element_potential_start(
        A::AbstractMatrix{T}, g::AbstractVector, b::AbstractVector, y0::AbstractVector;
        dead = Set{Int}(), maxit::Int = 200, tol::Float64 = 1.0e-10,
    ) where {T <: Real}
    m = size(A, 1)
    y = collect(float.(y0))
    alive = [i for i in eachindex(g) if !(i in dead)]
    Aa = Matrix(@view A[:, alive])
    ga = collect(float.(g[alive]))
    bscale = max(1.0, maximum(abs, b))

    xa = similar(ga)
    φ(yv) = begin
        u = -(transpose(Aa) * yv)
        # `exp` is clamped only against overflow; the line search below is what
        # keeps the iterates in a range where that clamp is never reached.
        -dot(b, yv) - sum(exp(clamp(u[j] - ga[j], -700.0, 300.0)) for j in eachindex(ga))
    end

    φ_cur = φ(y)
    for _ in 1:maxit
        u = -(transpose(Aa) * y)
        @inbounds for j in eachindex(ga)
            xa[j] = exp(clamp(u[j] - ga[j], -700.0, 300.0))
        end
        r = Aa * xa .- b
        maximum(abs, r) <= tol * bscale && break

        H = Aa * Diagonal(xa) * transpose(Aa)
        dmax = maximum(abs, diag(H))
        @inbounds for k in 1:m
            H[k, k] += max(dmax, 1.0) * 1.0e-12
        end
        δ = qr(H, ColumnNorm()) \ r

        # Backtracking on a concave function: an ascent step exists for small
        # enough `α`, so this cannot fail to make progress unless `∇φ = 0`.
        α = 1.0
        improved = false
        for _ in 1:60
            φ_try = φ(y .+ α .* δ)
            if isfinite(φ_try) && φ_try > φ_cur
                y .+= α .* δ
                φ_cur = φ_try
                improved = true
                break
            end
            α /= 2
        end
        improved || break
    end
    return y
end

# A callback returning dual numbers it captures: they cannot be stripped from
# here, so say so rather than fail on the first buffer of the solve.
function _refuse_captured_duals(name, v)
    eltype(v) <: ForwardDiff.Dual && throw(
        ArgumentError(
            "`$name` returns dual numbers it captures: pass the data being " *
                "differentiated through `params`, or solve on the values and lift the " *
                "answer with `dual_newton_tangent(...; primal)`.",
        ),
    )
    return nothing
end

"""
    dual_newton_solve(prob, b, x0; opts) -> (; x, y, q, active_phases, active, converged, kkt_error)

Solve `prob` for the right-hand side `b`, starting from `x0`.

`x0` supplies a neighborhood, not a feasible point: this is a Newton method, and
the intended use is to hand it the answer of an interior-point solve. Verify the
result with [`kkt_certificate`](@ref); for a convex problem that certificate is a
proof of **global** optimality.

With `opts.lp_fallback`, when no start converges and the problem has no unknown
parameters, the solve is tried once more from the vertex of the linear program
over the pure phases ([`lp_start`](@ref)): a neighborhood in the wrong assemblage
can hold the search, the vertex is in the right one.

# Two active sets

Over the **bound-constrained** variables, on the sign of `uᵢ − (gᵢ + hᵢ)`: a pure
phase is present exactly when that index vanishes, absent when it is negative.

Over the **mixing phases**, on Michelsen's tangent-plane measure, the one the
certificate applies ([`kkt_certificate`](@ref)): the log-sum-exp of
`uᵢ − gᵢ − ln γᵢ` over the live members of the phase at a trial composition of
it, a member whose component is absent from the budget left out. A solution
phase forms when that measure is positive, when a trial composition lies below
the tangent plane of the current state. That test is what a mixing phase
requires, since its members are never exactly absent while it exists and it has
no single saturation index.

Both admit one candidate per round, the most violated, and the sets visited are
recorded, which bounds the loop by the number of subsets and therefore
terminates. Admitting a batch feeds a cycle in which a variable is admitted,
driven negative, dropped and readmitted.
"""
function dual_newton_solve(
        prob::DualNewtonProblem, b::AbstractVector, x0::AbstractVector;
        opts::DualNewtonOptions = DualNewtonOptions(),
    )
    # A budget or data carrying dual numbers: solved on their values, with the
    # derivatives of the answer from the implicit-function theorem
    # (`_dual_newton_solve_ad`).
    (_is_dual_data(b) || _is_dual_data(prob.A) || _is_dual_data(prob.g) || _is_dual_data(prob.params)) &&
        return _dual_newton_solve_ad(prob, b, x0; opts)
    # Duals captured by the callbacks themselves cannot be stripped from here:
    # say so, rather than fail on the first buffer of the solve. `h` and, when
    # the problem has unknown parameters, `gq`, `cq` and `hq` alike.
    _refuse_captured_duals("h", current_h(prob, x0, prob.q0))
    if prob.nq > 0
        prob.gq === nothing || _refuse_captured_duals("gq", prob.gq(prob.q0, prob.params))
        _refuse_captured_duals("cq", prob.cq(x0, prob.q0, prob.params))
    end
    bv = Vector{Float64}(b)
    n0 = Vector{Float64}(x0)
    m = size(prob.A, 1)
    x_buf = zeros(Float64, length(prob.g))

    degenerate = _degenerate_conservation_rows(prob, bv)
    dead = _dead_variables(prob.A, degenerate)


    # One log-vector per phase, whether or not the phase is active: an inactive
    # phase keeps its last state, so readmitting it costs nothing.
    W = [Float64[log(max(n0[i], 1.0e-30)) for i in ph.members] for ph in prob.phases]
    for (k, ph) in pairs(prob.phases), (j, i) in pairs(ph.members)
        i in dead && (W[k][j] = W_FLOOR)
    end

    # A phase starts active if it is always present or if the guess holds it — and
    # never if every one of its members carries a vanished component.
    #
    # That last clause is not defensive tidying. A phase whose components are all
    # absent from the budget cannot exist, and its stationarity condition
    # `logsumexp(uᵢ − gᵢ) = 0` is evaluated over an empty set: every exponent is
    # `-Inf`, the log-sum-exp is `-Inf`, and the outer residual is `Inf` from the
    # first evaluation onwards. The admission test below already excludes such a
    # phase; the seeding did not, and against an interior-point warm start that
    # matters, because a barrier point holds even a dead species near `μ` rather
    # than at zero. On an LC³ budget, which carries no magnesium at all, that
    # seeded the M-S-H or hydrotalcite solution as present and the solve never
    # produced a finite residual.
    act_ph = Int[
        k for (k, ph) in pairs(prob.phases)
            if !all(i in dead for i in ph.members) &&
            (ph.always_present || sum(n0[i] for i in ph.members) > 1.0e-9)
    ]
    isempty(act_ph) && (act_ph = Int[1])
    # `refs[a]` is the reference member's amount for a phase with a solvent, and
    # the phase TOTAL for a mole-fraction phase — in both cases the quantity whose
    # logarithm the outer system carries.
    #
    # A mole-fraction phase that is always present starts from its own total: it
    # is present because a conservation row holds it, a surface site family whose
    # row fixes its total. Lifted to `PHASE_ADMISSION_SEED` like a phase being
    # admitted, the total of a family of 1e-9 mol started a thousand times too
    # high, the step changes `ln N` by at most one, and the search stopped with
    # 4e-9 mol of sites for a budget of 1e-9 (ChemistryLab, a sorbent in a
    # portlandite solution; correct from 1e-8 mol of sites up).
    refs = Float64[
        let ph = prob.phases[k], total = sum(n0[i] for i in ph.members)
            if !ph.mole_fraction
                max(n0[ph.members[ph.j_ref]], PHASE_ADMISSION_SEED)
            elseif ph.always_present
                max(total, floatmin(Float64))
            else
                max(total, PHASE_ADMISSION_SEED)
            end
        end for k in act_ph
    ]

    # The initial active set must already satisfy the phase rule, and reading it
    # off a threshold does not.
    #
    # `n0[i] > 1e-6` was the test, and against an interior-point warm start it is
    # no test at all: a barrier point at `μ` holds every ABSENT phase at
    # `sᵢ = μ/gᵢ`, which for `μ = 1e-6` is exactly the threshold. On an LC³ budget
    # that seeded 15 pure phases, 5 solid solutions on top, and 19 stationarity
    # conditions on 12 components — an active set with no solution, from the first
    # iteration, and no way back since the loop only ever rejects the variable it
    # just added.
    #
    # Candidates are therefore taken in order of decreasing amount — the phase
    # rule says at most `m` phases are present, and the abundant ones are the best
    # guess as to which — and admitted only while the set still supports a
    # solution. What the seeding leaves out is not lost: the saturation index
    # brings it back in below, by exchange.
    active = Int[]
    xB = Float64[]
    # Pinned variables go in first and unconditionally: they are determined by a
    # linear row, not admitted by a test, and the phase-rule check below would
    # reject a species sitting at zero.
    for i in prob.always_active
        i in active && continue
        push!(active, i)
        push!(xB, max(n0[i], 1.0e-12))
    end
    let cand0 = sort(
            [
                i for i in prob.idx_bounded
                    if n0[i] > 1.0e-6 && !(i in dead) && !(i in prob.always_active)
            ];
            by = i -> -n0[i],
        )
        for i in cand0
            push!(active, i)
            if _active_set_supports_a_solution(prob, active, act_ph)
                push!(xB, n0[i])
            else
                pop!(active)
            end
        end
    end

    # `y` from a WEIGHTED least-squares fit of the phase stationarity at the
    # guess. It holds for every phase member at the solution, but as a starting
    # point the fit must be driven by the variables that carry weight: an
    # interior-point answer is accurate on the major ones and can be a factor 1e5
    # out on a 1e-9 trace, and an unweighted fit lets those traces set `y`.
    fill!(x_buf, 0.0)
    for (a, k) in enumerate(act_ph), (j, i) in pairs(prob.phases[k].members)
        x_buf[i] = j == prob.phases[k].j_ref ? refs[a] : exp(W[k][j])
    end
    for (j, i) in enumerate(active)
        x_buf[i] = xB[j]
    end
    # The starting guess is built at the starting parameters, deliberately: the
    # Newton loop moves both together from there.
    h0 = current_h(prob, x_buf, prob.q0)
    # `y` from the stationarity of the phase members at the guess. At the
    # solution that condition holds for EVERY member, so any `m` independent
    # equations fix `y`; as a STARTING POINT the question is which of them the
    # guess gets right, and no single answer serves every problem.
    #
    #   * weighting by `√xᵢ` lets the solvent outweigh a trace by fifteen decades,
    #     so the fit becomes a statement about the solvent alone;
    #   * weighting equally lets a trace the guess got wrong by a factor 1e5 set
    #     the multipliers;
    #   * fitting on the largest members only is well conditioned but discards
    #     information the other two keep.
    #
    # Each is right somewhere and wrong elsewhere: measured on a cement replay,
    # the first certifies 36 instants of 40 and the third certifies the 4 it
    # misses while losing 34 of the others. So the solver TRIES them in turn and
    # keeps the first that converges. That is not a tuning knob — the acceptance
    # test is the KKT system itself, so a poor start can only cost time, never
    # correctness.
    all_members = vcat([prob.phases[k].members for k in act_ph]...)
    gh_fit = -(prob.g[all_members] .+ h0[all_members])
    A_all = Matrix(@view prob.A[:, all_members])

    # Candidates are ordered from the most informed by the caller's guess to the
    # least, and the loop stops at the first that converges. A caller replaying a
    # trajectory hands over a composition that is nearly the answer, and the fits
    # built from it succeed immediately; the element-potential solve at the end is
    # the cold-start device and is then never even run. Putting it first cost a
    # warm-started replay three digits of element balance — 5.7e-10 against
    # 1.4e-11 — for no gain, because the fits it displaced were the better guess.
    y_starts = Vector{Vector{Float64}}()

    let wgt = [sqrt(max(x_buf[i], 1.0e-30)) for i in all_members]
        push!(y_starts, qr(transpose(A_all * Diagonal(wgt)), ColumnNorm()) \ (gh_fit .* wgt))
    end
    let order = sortperm([x_buf[i] for i in all_members]; rev = true),
            nfit = min(length(all_members), max(3 * m, m + 4))
        sel = order[1:nfit]
        push!(y_starts, qr(transpose(A_all[:, sel]), ColumnNorm()) \ gh_fit[sel])
    end
    push!(y_starts, qr(transpose(A_all), ColumnNorm()) \ gh_fit)

    # Last candidate: the concave dual of the ideal problem, solved globally. See
    # `_element_potential_start` — it is the only one of these that knows the
    # element budget exists, and the only one that does not depend on the guess.
    let y_ls = copy(y_starts[end])
        for k in degenerate
            y_ls[k] = DEGENERATE_POTENTIAL
        end
        push!(
            y_starts,
            _element_potential_start(prob.A, prob.g, bv, y_ls; dead = dead),
        )
    end

    for yk in y_starts
        for k in degenerate
            yk[k] = DEGENERATE_POTENTIAL
        end
    end

    # Attempts are ranked by the KKT error they reach, not by the order they were
    # tried in. Keeping the first unless a later one CONVERGES throws away a better
    # answer whenever none converges: a start that lands at 1e8 was returned in
    # preference to one at 20, because neither had crossed the tolerance.
    best = nothing
    for y0 in y_starts
        out = _dual_newton_attempt(
            prob, bv, y0, W, act_ph, refs, active, xB, x_buf, dead, degenerate, m, opts,
        )
        if best === nothing || out.kkt_error < best.kkt_error
            best = out
        end
        out.converged && break
    end
    best.converged && return best

    # No start converged: once more from the vertex of the linear program over
    # the pure phases. A start that carries the wrong assemblage can hold the
    # search there: from it, the phase that should enter is admitted, the inner
    # inversion fails at once, the entrant is rejected, and the one that should
    # leave never does. Measured on a Portland paste at the instant hydrogarnet
    # gives way to monosulfate (Lerch and Ford's cement c13, 4.2 h at 23.9 °C):
    # from the partition of the instant before, every start ends at a KKT error
    # of 1.5e-4 or worse; from the vertex, which holds the new assemblage, the
    # solve certifies at once. The vertex is floored at `LP_FALLBACK_FLOOR`: the
    # amounts it leaves at zero must sit below anything the search would read as
    # present (from 1e-9 the same case fails again). Only a solve that did not
    # converge pays for this, and a problem with unknown parameters is left
    # alone: the program knows nothing of `Aq`. Off by default (see
    # `DualNewtonOptions`).
    if opts.lp_fallback && prob.nq == 0
        lp = lp_start(prob, bv)
        if lp.status === :optimal
            opts.verbose && @info "dual-newton: no start converged; from the vertex of the linear program"
            again = dual_newton_solve(
                prob, bv, max.(lp.x, LP_FALLBACK_FLOOR); opts = _without_lp_fallback(opts),
            )
            (again.converged || again.kkt_error < best.kkt_error) && return again
        end
    end
    return best
end

# ── the exact Jacobian of the outer residual ─────────────────────────────────
#
# The outer residual is `R(v) = F(v, W(v))`, the compositions `W` being what the
# inner inversion recovers from `v`: `G(v, W) = 0`. Where the inversion has
# converged, the implicit-function theorem gives its Jacobian exactly,
#
#     dR/dv = F_v − F_W G_W⁻¹ G_v,
#
# with every partial derivative by forward-mode differentiation over `z = (v, w)`,
# `w` the log-amounts the inversion determines.

# The unknowns of the inversion at a converged state: `(a, k, j, i, mf)` for each
# member it determines, `a` the phase's place in `act_ph`, `mf` whether the phase
# is a mole-fraction one. A member at a bound it cannot leave is not one of them.
function _inner_unknowns(prob, W, v, act_ph, dead)
    m = size(prob.A, 1)
    nph = length(act_ph)
    y = v[(nph + 1):(nph + m)]
    out = Tuple{Int, Int, Int, Int, Bool}[]
    for (a, k) in enumerate(act_ph)
        ph = prob.phases[k]
        for (j, i) in enumerate(ph.members)
            i in dead && continue
            if ph.mole_fraction
                W[k][j] <= W_FLOOR + 1.0e-9 && continue
                push!(out, (a, k, j, i, true))
            else
                j == ph.j_ref && continue
                (W[k][j] <= W_FLOOR + 1.0e-9 || W[k][j] >= W_CEIL - 1.0e-9) && continue
                push!(out, (a, k, j, i, false))
            end
        end
    end
    return out
end

# `[G; F]` at `z = (v, w)`, in the element type of `z`.
function _implicit_system(prob, z, unk, W, act_ph, active, bv; dead, degenerate)
    Tz = eltype(z)
    m = size(prob.A, 1)
    nph = length(act_ph)
    na = length(active)
    nq = prob.nq
    Nv = nph + m + na + nq
    y = z[(nph + 1):(nph + m)]
    xB = z[(nph + m + 1):(nph + m + na)]
    q = z[(nph + m + na + 1):Nv]

    x = zeros(Tz, length(prob.g))
    for k in act_ph, (j, i) in enumerate(prob.phases[k].members)
        x[i] = exp(W[k][j])
    end
    for (a, k) in enumerate(act_ph)
        ph = prob.phases[k]
        ph.mole_fraction || (x[ph.members[ph.j_ref]] = exp(z[a]))
    end
    for (p, (_, _, _, i, _)) in enumerate(unk)
        x[i] = exp(z[Nv + p])
    end
    for (j, i) in enumerate(active)
        x[i] = xB[j]
    end

    g = nq == 0 ? prob.g : current_g(prob, q)
    hv = current_h(prob, x, q)
    u = -(transpose(prob.A) * y)
    cqv = nq == 0 ? Tz[] : prob.cq(x, q, prob.params)
    # The element type of the system: that of the unknowns, or of the data when
    # it carries the duals of a caller differentiating the answer, the
    # constraint's equations included (a prescribed target they capture).
    T = promote_type(Tz, eltype(g), eltype(hv), eltype(u), eltype(prob.A), eltype(bv), eltype(cqv))

    # The exponents `u − g − lnγ` of each mole-fraction phase and their
    # log-sum-exp, over its live members.
    lZ = Dict{Int, T}()
    dexp = Dict{Tuple{Int, Int}, T}()
    for (a, k) in enumerate(act_ph)
        ph = prob.phases[k]
        ph.mole_fraction || continue
        N = exp(z[a])
        ds = T[]
        for (j, i) in enumerate(ph.members)
            i in dead && continue
            lnγ = x[i] > 0 ? hv[i] - log(x[i] / N) : zero(T)
            d = u[i] - g[i] - lnγ
            dexp[(k, j)] = d
            push!(ds, d)
        end
        M = maximum(ForwardDiff.value, ds)
        lZ[k] = M + log(sum(exp(d - M) for d in ds))
    end

    G = T[
        mf ? z[Nv + p] - (z[a] + dexp[(k, j)] - lZ[k]) : (u[i] - g[i]) - hv[i]
            for (p, (a, k, j, i, mf)) in enumerate(unk)
    ]
    R_ref = T[
        let ph = prob.phases[k]
            ph.mole_fraction ? lZ[k] :
                (ir = ph.members[ph.j_ref]; g[ir] + hv[ir] - u[ir])
        end for k in act_ph
    ]
    Rb = nq == 0 ? prob.A * x .- bv : prob.A * x .+ prob.Aq * q .- bv
    Rb = T.(Rb)
    for k in degenerate
        Rb[k] = y[k] - DEGENERATE_POTENTIAL
    end
    Rs = T[g[i] + hv[i] - u[i] for i in active]
    Rq = T.(cqv)
    return vcat(G, R_ref, Rb, Rs, Rq)
end

# The inner unknowns the inversion determines, given `G_W`: a member whose own
# amount does not enter its equation (a model that floors a vanishing activity,
# below its floor) is not one of them. It is held, its equation dropped and its
# amount a constant.
_determined(Gw) = [p for p in axes(Gw, 1) if abs(Gw[p, p]) > 1.0e-10]

"""
    _outer_jacobian(prob, v, W, act_ph, active, bv; dead, degenerate) -> Matrix

The Jacobian of the outer residual at `v`, the inversion having converged at `W`,
by the implicit-function theorem and forward-mode differentiation. The callbacks
of the problem (`h`, and `gq`, `hq`, `cq` when it has parameters) must accept
dual numbers.
"""
function _outer_jacobian(prob, v, W, act_ph, active, bv; dead, degenerate)
    unk = _inner_unknowns(prob, W, v, act_ph, dead)
    z0 = vcat(v, Float64[W[k][j] for (_, k, j, _, _) in unk])
    Jz = ForwardDiff.jacobian(z -> _implicit_system(prob, z, unk, W, act_ph, active, bv; dead, degenerate), z0)
    Nv = length(v)
    nG = length(unk)
    Gv = Jz[1:nG, 1:Nv]
    Gw = Jz[1:nG, (Nv + 1):end]
    Fv = Jz[(nG + 1):end, 1:Nv]
    Fw = Jz[(nG + 1):end, (Nv + 1):end]
    nG == 0 && return Fv
    keep = _determined(Gw)
    Gv, Gw, Fw = Gv[keep, :], Gw[keep, keep], Fw[:, keep]
    isempty(keep) && return Fv
    F = lu(Gw; check = false)
    # A singular `G_W` (a composition the inversion does not determine) gets the
    # least-squares derivative.
    return Fv .- Fw * (issuccess(F) ? F \ Gv : qr(Gw, ColumnNorm()) \ Gv)
end

"""
    _swept_jacobian(prob, v, W_ref, act_ph, active, bv; dead, degenerate,
                    inner_maxit, max_fall) -> Matrix

The Jacobian of the outer residual as it is evaluated where the inversion has
not converged: the sweeps it runs from `W_ref`, the warm start every candidate
of the line search is also evaluated from, differentiated by forward mode. The
implicit-function Jacobian (`_outer_jacobian`) is that of the converged
inversion, and differs from the derivative of the residual the Newton actually
sees by as much as the inversion is short of convergence: measured on the
coupled hydration of a CEM I over three hours, where the inversion ends
unconverged at most iterations, it took 5007 Newton iterations where this takes
the number the difference quotients of 0.7.4 did, on the same trajectory.
"""
function _swept_jacobian(prob, v, W_ref, act_ph, active, bv; dead, degenerate, inner_maxit, max_fall)
    f = function (vv)
        T = eltype(vv)
        Wd = [Vector{T}(w) for w in W_ref]
        xd = zeros(T, length(prob.g))
        return _outer_residual(
            prob, vv, Wd, act_ph, active, bv, xd;
            dead, degenerate, inner_maxit, max_fall,
        )
    end
    return ForwardDiff.jacobian(f, v)
end

# ── the balance, judged row by row ────────────────────────────────────────────
#
# A balance row is judged against what it holds, `Σⱼ |Aₖⱼ xⱼ| + Σₗ |Aqₖₗ qₗ|`,
# and in moles once that exceeds a mole: `|rₖ| ≤ tol·min(scaleₖ, 1)`. A trace is
# then held to its own amount, as PHREEQC and GEMS hold a mass balance to its
# element total, and a large row to the absolute tolerance it always had, so no
# answer is accepted that was refused before. Judged in moles alone, a trace of
# 1e-9 mol could be 10 % wrong and pass.
#
# The scale never falls below a floor: `eps²` of the budget for a row that has
# one, so that a row whose carriers have all vanished is refused, and `eps` of
# the largest budget for a row whose budget is zero within rounding. The same
# scales weight the rows of the Newton step.
#
# Only a row with a budget is judged relative to what it holds. A row whose
# budget is zero within rounding has no total to be a fraction of (the electron
# row of a redox pair, a coupled site family, the extents of a kinetic step), and
# is judged in moles, as a degenerate row is. Judged against its floor, the
# electron row of a cement paste, whose carriers held 8e-15 mol, read 1.24e-24
# mol as 1.5e-10 of it, and held the Newton of a certified replay for 5033 of its
# 6283 evaluations: three and a half times the time, for nothing.

function _balance_scales(prob, x, q, b, degenerate)
    m = size(prob.A, 1)
    bscale = max(1.0, maximum(abs, b; init = 0.0))
    scales = ones(m)
    for k in 1:m
        k in degenerate && continue
        s = 0.0
        for j in axes(prob.A, 2)
            s += abs(prob.A[k, j] * _primal_value(x[j]))
        end
        for l in eachindex(q)
            s += abs(prob.Aq[k, l] * _primal_value(q[l]))
        end
        floor_k = abs(b[k]) > eps() * bscale ? eps()^2 * abs(b[k]) : eps() * bscale
        scales[k] = max(s, floor_k)
    end
    return scales
end

# The scale a balance row is judged on: what it holds, for a row with a
# budget; one, in moles, for a row whose budget is zero within rounding.
function _judged_scales(scales, b)
    bscale = max(1.0, maximum(abs, b; init = 0.0))
    return [abs(b[k]) > eps() * bscale ? scales[k] : 1.0 for k in eachindex(scales)]
end

# The worst row of `R`, its balance rows (from `off + 1`) divided by
# `min(scale, 1)` and every other row as it is.
function _judged_residual(R, scales, off)
    worst = 0.0
    for i in eachindex(R)
        r = abs(_primal_value(R[i]))
        k = i - off
        (1 <= k <= length(scales)) && (r /= min(scales[k], 1.0))
        worst = max(worst, r)
    end
    return worst
end

# A basis exchange: the first incumbent among the positions `candidates`, in
# order of increasing amount, whose removal lets the active set support a
# solution, removed from `active` and from its amounts `xB`. Both are returned
# unchanged when no removal does.
function _exchange_incumbent(prob, active, xB, act_ph, candidates)
    order = sort(candidates; by = j -> xB[j])
    without(v, j) = v[setdiff(eachindex(v), [j])]
    k = findfirst(j -> _active_set_supports_a_solution(prob, without(active, j), act_ph), order)
    return k === nothing ? (active, xB) : (without(active, order[k]), without(xB, order[k]))
end

"""
    _newton_on_active_set!(W, prob, v, xB, act_ph, active, bv, x_buf, dead, degenerate, m, opts)
        -> (v, inner_ok)

The Newton iteration of the outer system on one active set, from `v`: the
residual weighted row by row, the exact Jacobian where the inner inversion
has converged (the swept one elsewhere), the step capped, and the two-pass
line search. `W`, the log-compositions of the phases, is updated in place.
Returns the last accepted `v` and whether the residual met `opts.tol`.
"""
function _newton_on_active_set!(W, prob, v, xB, act_ph, active, bv, x_buf, dead, degenerate, m, opts)
    nph = length(act_ph)
    na = length(active)
    nq = prob.nq
    inner_ok = false
    inner_resid = Ref(Inf)
    # Which phases' own inversion found a composition at the iterate: what a
    # trial that finds none is judged against (`_invert_phases!`).
    composed = trues(length(prob.phases))
    # The composition at the iterate, put back when no step is accepted: the
    # caller reads `x_buf` for the saturation indices of the species held at
    # their bound, and a rejected trial leaves its own composition there. On a
    # cement paste of ChemistryLab's thesis corpus, about a hundred line
    # searches end so over the paste's run, three in four on a trial whose
    # inversion had run away to the ceiling.
    x_iterate = similar(x_buf)
    for _ in 1:(opts.maxit)
        R = _outer_residual(
            prob, v, W, act_ph, active, bv, x_buf;
            dead, degenerate, resid = inner_resid, inner_maxit = opts.inner_maxit,
            max_fall = opts.inner_fall_bound, composed,
        )
        copyto!(x_iterate, x_buf)
        res = maximum(abs, R)
        # Read before the Jacobian reuses `x_buf`, and the weights of the step
        # below: each balance row on the scale of what it holds.
        scales = _balance_scales(
            prob, x_buf, @view(v[(nph + m + na + 1):(nph + m + na + nq)]), bv, degenerate,
        )
        judged = _judged_residual(R, _judged_scales(scales, bv), nph)
        if opts.verbose
            # Split by block: the three carry different units — log-activities
            # for the phase and stationarity rows, moles for the balance — and
            # a single maximum says nothing about which of them is stuck.
            rp = nph == 0 ? 0.0 : maximum(abs, @view R[1:nph])
            rb = maximum(abs, @view R[(nph + 1):(nph + m)])
            rs = na == 0 ? 0.0 : maximum(abs, @view R[(nph + m + 1):(nph + m + na)])
            @info "dual-newton" res judged res_phase = rp res_balance = rb res_stat = rs nph na
        end
        if judged <= opts.tol
            inner_ok = true
            break
        end

        N = length(v)
        W_ref = [copy(w) for w in W]
        # Each balance row weighted by the scale it is judged on, what it
        # currently holds (`scales`, read above), for the factorization
        # below. The rank of a pivoted QR is decided against its
        # largest pivot, and the row of a trace component carries the
        # derivatives of amounts as small as its carriers: a budget of 1e-9
        # mol whose carriers sat at 1e-16 gave its potential a column of
        # 1e-16 beside entries of order a hundred, below the rank threshold,
        # so the step left that potential where it was and the Newton stopped
        # with the row unmet. On a square system the weights do not change
        # the step; they change which directions count as there.
        #
        # The floor tells the two kinds of small row apart. A row with a
        # budget is resolved down to carriers `eps²` of it, which also keeps
        # the weighted residual finite. A row whose budget is zero within
        # rounding keeps the floor of that rounding, so that a direction no
        # species carries (a redox potential held by amounts of 1e-305) stays
        # as absent as it was.
        wR = ones(length(R))
        for k in 1:m
            wR[nph + k] = 1.0 / scales[k]
        end
        # Exact, by the implicit-function theorem over the inversion where it
        # has converged (`_outer_jacobian`), and by forward mode through its
        # sweeps where it has not (`_swept_jacobian`): in both cases the
        # derivative of the residual the line search then evaluates.
        J = inner_resid[] <= opts.inner_tol ?
            _outer_jacobian(prob, v, W, act_ph, active, bv; dead, degenerate) :
            _swept_jacobian(
                prob, v, W_ref, act_ph, active, bv; dead, degenerate,
                inner_maxit = opts.inner_maxit, max_fall = opts.inner_fall_bound,
            )
        all(isfinite, J) || break

        δ = qr(wR .* J, ColumnNorm()) \ (-(wR .* R))

        α = 1.0
        for a in 1:nph
            abs(δ[a]) > 1.0 && (α = min(α, 1.0 / abs(δ[a])))
        end
        # The potential of a species, `−Aᵀy`, moves by no more per step than
        # the inner inversion lets a log-amount rise in one sweep. The
        # balance of a component is a sum of exponentials of its potential,
        # and from below its budget the linearization overshoots by the
        # ratio of the two: carriers twenty-one orders of magnitude short ask
        # for a step of 1e21 in a potential that has to move by forty-eight,
        # which no backtracking down to 2⁻⁴⁰ brings within reach, and the
        # Newton stopped there with that balance unmet.
        dmax = 0.0
        for j in axes(prob.A, 2)
            j in dead && continue
            dj = 0.0
            for k in 1:m
                dj += prob.A[k, j] * δ[nph + k]
            end
            dmax = max(dmax, abs(dj))
        end
        dmax > W_MAX_RISE && (α = min(α, W_MAX_RISE / dmax))
        for j in 1:na
            dj = δ[nph + m + j]
            if dj < 0 && xB[j] + α * dj < 0
                α = min(α, -0.9 * xB[j] / dj)
            end
        end

        # A step is accepted on two conditions, not one.
        #
        # The decrease is the obvious one. The second is that the candidate's
        # INNER solve converged, and it is what makes the test mean anything:
        # the inner fixed point is warm-started, so until it has converged its
        # answer — and hence `R_t` — depends on the warm start it was handed.
        # Accepting such a step compares a residual the next iteration will
        # not reproduce, and it does not reproduce it: measured on a CEM I
        # with eight solid solutions, an accepted "descent" step was followed
        # by a residual of 4.2e17 where the step itself had reported 60.5, and
        # the active-set round recorded that 4.2e17 as the state's KKT error.
        # With the gate below the same problem certifies to 1.1e-14, and stops
        # depending on the last bit of its own input.
        accepted = false
        cand_resid = Ref(Inf)
        α0 = α
        # Two passes, and the order is the whole point.
        #
        # The first asks for a step that both decreases the residual and whose
        # inner solve converged; the second drops the second condition. An
        # inner solve that has not converged leaves `W` still moving, so the
        # `R_t` it reports is not the residual the next iteration will measure
        # at the same `v` — measured on a CEM I with eight solid solutions, an
        # accepted step reporting 60.5 was followed by 4.2e17. Preferring a
        # converged candidate removes that, and the problem stops depending on
        # the last bit of its own input.
        #
        # The second pass is not a concession: an inner iteration whose `h`
        # does not depend on the composition has nothing to solve and can
        # never report convergence, and on such a problem the first pass would
        # refuse every step and the solve would stall at its starting point.
        #
        # With `lenient_line_search`, the first pass is not asked for what
        # the current point does not have. Where the inner solve has not
        # converged at `v` itself, shortening the step does not make it
        # converge at `v + αδ`: the first pass then accepts what the second
        # would, the first step that decreases the residual, instead of
        # refusing forty candidates to reach the same one. Measured on the
        # coupled hydration of a CEM I over three hours, that was 392 044
        # inner solves spent in the first pass for 189 steps accepted there.
        #
        # Nor is it asked of an iterate whose potentials hold no composition
        # for a phase that inverts itself, lenient or not: that phase was
        # swept, and so are its trials (`composed`), none of which reports a
        # converged inversion unless it holds a composition again. Asked
        # anyway, the first pass refused forty swept trials, each swept to
        # the stall rule, before the second accepted the first that
        # decreased: 32 cement pastes then took 505 s instead of 349 s,
        # seven of them two to three times longer.
        current_converged = (!opts.lenient_line_search || inner_resid[] <= opts.inner_tol) &&
            all(k -> composed[k], act_ph)
        # The decrease asked for is that of the worst row or, where that one
        # cannot decrease, Armijo's on `‖R‖²` with the worst row not growing.
        # The step is the least-squares one, a descent direction of `‖R‖²`
        # and not always of `max|Rᵢ|`: on an active set holding more
        # stationarity conditions than there are multipliers, the rows it
        # cannot satisfy stay where they are, and a test on the worst row
        # alone refuses the step that restores the element balance.
        #
        # The slope of `‖R‖²` along the least-squares step is `−2‖Jδ‖²`,
        # the part of the residual the linearization can remove, so that is
        # what the decrease is measured against. Below the tolerance it
        # predicts no change worth a step: the iterate is then the
        # least-squares point of this active set.
        # In the metric the step was taken in, the balance rows weighted.
        φ = sum(abs2, wR .* R)
        pred = sum(abs2, wR .* (J * δ))
        v_t = similar(v)
        W_t = [similar(w) for w in W_ref]
        for strict in (true, false)
            α = α0
            # Trials whose multipliers hold no composition (an inversion that
            # ran away, `_invert_phases!`) say nothing of the residual. A pass
            # that met only those leaves the other pass nothing to meet either,
            # the inversion not depending on the pass; and twenty in a row,
            # the step cut by a million, end the pass. Measured under the
            # limiting law past its range, nine trials in ten were such, and
            # each pass ran its forty.
            only_runaway = true
            runaway_streak = 0
            for _ in 1:40
                v_t .= v .+ α .* δ
                foreach(copyto!, W_t, W_ref)
                R_t = _outer_residual(
                    prob, v_t, W_t, act_ph, active, bv, x_buf;
                    dead, degenerate, resid = cand_resid,
                    inner_maxit = opts.inner_maxit, max_fall = opts.inner_fall_bound, trial = true,
                    composed,
                )
                if isinf(cand_resid[])
                    runaway_streak += 1
                    runaway_streak >= 20 && break
                    α /= 2
                    continue
                end
                only_runaway = false
                runaway_streak = 0
                mx = maximum(abs, R_t)
                decrease = mx < res || (
                    pred > opts.tol^2 && mx <= res * (1 + 1.0e-12) &&
                        sum(abs2, wR .* R_t) <= φ - 2.0e-4 * α * pred
                )
                ok = decrease &&
                    (!strict || !current_converged || cand_resid[] <= opts.inner_tol)
                if ok
                    v = v_t
                    for kk in eachindex(W)
                        W[kk] .= W_t[kk]
                    end
                    inner_resid[] = cand_resid[]
                    accepted = true
                    break
                end
                α /= 2
            end
            # Where the first pass asked nothing of the inversion, the second
            # would evaluate the same trials and judge them alike.
            (accepted || only_runaway || !current_converged) && break
        end
        if !accepted
            copyto!(x_buf, x_iterate)
            break
        end

        xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]
        # A pinned variable may legitimately pass through small values, so the
        # early break looks only at the ones an active set actually decides.
        let free = [j for j in 1:na if !(active[j] in prob.always_active)]
            !isempty(free) && minimum(@view xB[free]) < opts.si_tol && break
        end
        nph > 0 && minimum(@view v[1:nph]) < log(opts.si_tol) && break
    end
    return v, inner_ok
end

"""
    _dual_newton_attempt(...) -> (; x, y, q, active_phases, active, converged)

One run of the two-level Newton from a given set of multipliers. Called once per
starting point by [`dual_newton_solve`](@ref).
"""
function _dual_newton_attempt(
        prob, bv, y_init, W0, act_ph0, refs0, active0, xB0, x_buf, dead, degenerate, m, opts,
    )
    # The sets, amounts and parameters below are rebound as the search moves, and
    # a comprehension or a closure that captured them directly would box them —
    # every later use dispatched at run time. Each one is therefore handed over
    # through a `let`, which captures a binding that is never reassigned.
    y = copy(y_init)
    W = [copy(w) for w in W0]
    act_ph = copy(act_ph0)
    refs = copy(refs0)
    active = copy(active0)
    xB = copy(xB0)
    nq = prob.nq
    q = copy(prob.q0)

    converged = false
    seen = Set{Tuple{Vector{Int}, Vector{Int}}}()
    # Candidates whose admission was tried and left the stationarity block
    # unsatisfiable. Two bound-constrained variables both declared stationary
    # over-determine `y` whenever their formulas are dependent modulo the span of
    # the mixing phases: the least-squares step is then the best available and the
    # residual simply cannot reach zero. Observed on a cement at six hours as a
    # residual falling to 1.05, jumping to 18.9 on the admission, and stalling
    # there. Rejecting the candidate permanently keeps the outer loop finite and
    # lets the next one be tried.
    rejected = Set{Int}()
    drop_ph_prev = Int[]
    last_added = 0

    # The active-set search is a DESCENT method on the outer residual.
    #
    # Each move — admit the most violated candidate, release the most violated
    # incumbent — is a guess, and a guess that makes the residual worse has to be
    # undone, not built upon. Without that the search wanders: on an LC³
    # equilibrium the residual went 16.04 → 8.05 → 16.04 → 514 and stalled there,
    # having passed through and abandoned its best state. Accepting only
    # improvements makes the sequence of visited sets strictly decreasing in
    # residual, hence finite, and the answer is the best set found rather than the
    # last one tried.
    best_res = Inf
    best_state = nothing
    # The tangent-plane measure of each absent phase within a round, `NaN` until
    # asked for: three steps of a round ask for it at the same multipliers and
    # composition, and each evaluation runs up to `maxit` substitution sweeps.
    admission = fill(NaN, length(prob.phases))

    for _ in 1:(opts.max_active_updates)
        nph = length(act_ph)
        na = length(active)
        v = vcat([log(r) for r in refs], y, xB, q)
        v, inner_ok = _newton_on_active_set!(
            W, prob, v, xB, act_ph, active, bv, x_buf, dead, degenerate, m, opts,
        )

        refs = let v = v
            [exp(v[a]) for a in 1:nph]
        end
        y = v[(nph + 1):(nph + m)]
        xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]
        q = nq == 0 ? Float64[] : v[(nph + m + na + 1):(nph + m + na + nq)]

        u = -(transpose(prob.A) * y)
        hv = current_h(prob, x_buf, q)
        si = u .- (current_g(prob, q) .+ hv)
        fill!(admission, NaN)

        # Record this set if it is the best seen, measured by the KKT error of the
        # WHOLE problem — not by the residual of the subproblem this set defines.
        #
        # The distinction decides the search. An active set that omits a phase the
        # solution needs still solves its own equations exactly: the outer residual
        # goes to 1e-12 while the omitted phase sits absent and supersaturated by
        # ten RT. Ranking states on that residual therefore rewards leaving phases
        # out. The measure below is the one the certificate applies — stationarity,
        # element balance, and the worst violation among the phases held absent —
        # so descending it descends the distance to a KKT point.
        let res_outer = _judged_residual(
                _outer_residual(
                    prob, v, W, act_ph, active, bv, x_buf;
                    dead, degenerate, inner_maxit = opts.inner_maxit,
                    max_fall = opts.inner_fall_bound,
                ),
                _judged_scales(_balance_scales(prob, x_buf, q, bv, degenerate), bv), nph,
            )
            viol = 0.0
            for i in prob.idx_bounded
                (i in active || i in dead) && continue
                viol = max(viol, si[i])
            end
            for k in eachindex(prob.phases)
                k in act_ph && continue
                all(i in dead for i in prob.phases[k].members) && continue
                viol = max(viol, _admission_measure!(admission, prob, k, u, x_buf, q, dead))
            end
            kkt_err = max(res_outer, viol)
            opts.verbose && @info "active-set round" kkt_err res_outer viol nph na inner_ok
            if kkt_err < best_res * (1 - 1.0e-9)
                best_res = kkt_err
                # `v` carries refs, `y` and the bounded amounts together, so the
                # state is that vector plus the sets it is indexed by.
                best_state = (
                    copy(act_ph), copy(active), copy(v), [copy(w) for w in W], viol,
                )
            end
        end

        # ── bound-constrained variables ──
        # Complementarity, not just the amount.
        #
        # For a bound-constrained variable the KKT conditions are `xᵢ ≥ 0`,
        # `sᵢ ≤ 0` and `xᵢ sᵢ = 0` with `sᵢ = uᵢ − gᵢ − hᵢ`. Testing only
        # `xᵢ → 0` catches one half: a variable held ACTIVE while it is
        # UNDERSATURATED — `sᵢ` strictly negative — violates complementarity just as
        # plainly, and no amount of Newton iteration will repair it, because its own
        # equation `sᵢ = 0` is the one that cannot hold. It has to leave.
        #
        # Measured on an LC³ equilibrium at a quarter of full reaction, the search
        # settled with nothing supersaturated, the element balance at 4.5e-2 and a
        # stationarity residual of 9.86 carried entirely by such a phase: present,
        # and undersaturated by ten RT.
        # …and only on a point that solves the current subproblem. While the inner
        # Newton is still working, `sᵢ` on an active variable is a transient, not a
        # violation, and dropping on it removes phases that were on their way to
        # stationarity.
        drop = let active = active, xB = xB, inner_ok = inner_ok
            [
                j for j in eachindex(xB)
                    if !(active[j] in prob.always_active) &&
                    (
                        xB[j] < opts.si_tol ||
                        (inner_ok && si[active[j]] < -max(opts.si_tol, opts.tol))
                    )
            ]
        end

        # An admission that does not converge is not undone by rejecting the
        # entrant for good: that would read the failure backwards.
        #
        # The inner Newton breaks out the moment an active variable falls below
        # the bound (`minimum(xB) < si_tol`), and that is not a failed admission
        # — it is the departing variable announcing itself. The `drop` list above
        # already holds it, and letting the normal path remove it while the
        # entrant stays IS the active-set exchange. Treating it as the entrant's
        # fault is what left a cement with ettringite rejected and monosulphate
        # in its place: the two compete for the same sulfate, so admitting one
        # necessarily drives the other out, and the solver then converged —
        # beautifully, to 1e-12 — onto an assemblage in which ettringite was
        # absent and supersaturated by 14.8 RT.
        #
        # So the entrant is only reconsidered when the Newton failed with NOTHING
        # leaving, which is the genuine over-determination this guard was written
        # for: two bound variables declared stationary whose formulas are
        # dependent modulo the mixing phases, where the residual cannot reach
        # zero at all.
        # A stalled Newton with nothing leaving and nothing newly admitted means the
        # active set itself cannot be satisfied, and without what follows the
        # loop has no way out of that: `drop` tests only the AMOUNTS, so a phase
        # that is held active while its stationarity `uᵢ = gᵢ` is unreachable
        # stays for ever.
        # Measured on an LC³ equilibrium the stationarity residual sat at 12.5 with
        # the element balance at 0.02 — the least-squares step sacrificing the one
        # to hold the other, which is what an inconsistent system looks like.
        #
        # The variable to release is the one whose own equation carries the
        # residual: removing that equation is precisely what lets the rest be
        # satisfied, and `si[i] = uᵢ − gᵢ − hᵢ` IS the residual of active variable
        # `i`. This is the leaving rule of an active-set method — the entering rule
        # is the most violated candidate, the leaving rule the most violated
        # incumbent — and `seen` still bounds the search.
        if !inner_ok && isempty(drop) && last_added == 0 && !isempty(active)
            releasable, worst = let active = active
                r = [j for j in eachindex(active) if !(active[j] in prob.always_active)]
                r, isempty(r) ? 0 : r[argmax([abs(si[active[j]]) for j in r])]
            end
            if worst != 0 && abs(si[active[worst]]) > opts.tol
                push!(rejected, active[worst])
                active = active[setdiff(eachindex(active), [worst])]
                xB = xB[setdiff(eachindex(xB), [worst])]
            end
        end

        if !inner_ok && isempty(drop) && last_added != 0 && last_added in active
            push!(rejected, last_added)
            j = findfirst(==(last_added), active)
            if j !== nothing
                active = active[setdiff(eachindex(active), [j])]
                xB = xB[setdiff(eachindex(xB), [j])]
            end
            last_added = 0
        end

        # A veto is only ever valid in the context that produced it. Once the
        # active set has moved for any other reason, a variable refused earlier
        # deserves another hearing — otherwise one unlucky ordering decides the
        # assemblage permanently.
        (isempty(drop) && isempty(drop_ph_prev)) || empty!(rejected)

        cand = let active = active
            [
                i for i in prob.idx_bounded
                    if !(i in active) && !(i in dead) && !(i in rejected) && si[i] > opts.si_tol
            ]
        end

        # What was vetoed is still measured. A rejected variable that remains
        # supersaturated is a violated KKT condition, and the run must not be
        # reported as converged just because the candidate list was filtered.
        veto_violation = let active = active
            any(i -> !(i in active) && !(i in dead) && si[i] > opts.si_tol, rejected)
        end

        # ── mixing phases ──
        drop_ph = let refs = refs
            [
                a for (a, k) in enumerate(act_ph)
                    if !prob.phases[k].always_present && refs[a] < opts.si_tol
            ]
        end
        # Where the Newton stalled with nothing leaving, the most undersaturated
        # phase is the incumbent to release, as the bounded variable carrying the
        # residual is above: on an active set breaking the phase rule, the
        # stationarity of its reference cannot be met at any amount, so no step
        # reduces it. Waiting for its amount to fall below `si_tol` relied on the
        # noise of a difference quotient, which drove it down by accident.
        #
        # A present phase is judged at the composition the inversion gave it,
        # which is the one its own stationarity selects at these multipliers, and
        # not by the tangent-plane search that admits an absent phase: that search
        # starts from the corners and can return a local maximum below zero for a
        # phase inside a miscibility gap, which would then be released although
        # the answer needs it.
        if !inner_ok && isempty(drop) && isempty(drop_ph) && last_added == 0
            gq_now = current_g(prob, q)
            hv_now = current_h(prob, x_buf, q)
            under = [
                (a, _phase_saturation(prob, k, u, hv_now, x_buf, gq_now, dead))
                    for (a, k) in enumerate(act_ph) if !prob.phases[k].always_present
            ]
            filter!(t -> t[2] < -max(opts.si_tol, opts.tol), under)
            isempty(under) || push!(drop_ph, first(under[argmin([t[2] for t in under])]))
        end
        cand_ph = let act_ph = act_ph, q = q
            [
                k for k in eachindex(prob.phases)
                    if !(k in act_ph) && !all(i in dead for i in prob.phases[k].members) &&
                    _admission_measure!(admission, prob, k, u, x_buf, q, dead) > opts.si_tol
            ]
        end

        if isempty(drop) && isempty(cand) && isempty(drop_ph) && isempty(cand_ph)
            converged = inner_ok && !veto_violation
            break
        end

        if !isempty(drop)
            keep = setdiff(eachindex(xB), drop)
            active = active[keep]
            xB = xB[keep]
        end
        if !isempty(drop_ph)
            keep = setdiff(eachindex(act_ph), drop_ph)
            act_ph = act_ph[keep]
            refs = refs[keep]
        end
        if !isempty(cand)
            i_best = cand[argmax([si[i] for i in cand])]
            push!(active, i_best)
            push!(xB, 1.0e-9)
            # Admitting a variable may leave the active set unable to support a
            # solution at all, either because the stationarity capacity `m` is
            # already spent or because the entrant's composition is a combination
            # of those already active. Neither is a reason to refuse it — it is
            # violated, so it belongs — but one of the incumbents has to leave.
            #
            # Removal candidates are tried in order of increasing amount: the
            # smallest is the one closest to leaving on its own and the one whose
            # removal perturbs the primal least. Where the entrant is DEPENDENT on
            # the active set, only a variable it is dependent with restores the
            # rank, and trying them in turn finds it. This is a basis exchange, and
            # the choice among admissible incumbents is a heuristic — what is not
            # heuristic is that the set must satisfy the rank and capacity
            # conditions, since otherwise no `y` exists. Termination is unaffected:
            # `seen` records the active sets visited and there are finitely many.
            if !_active_set_supports_a_solution(prob, active, act_ph)
                active, xB = _exchange_incumbent(prob, active, xB, act_ph, 1:(length(active) - 1))
            end
            # If nothing worked the entrant itself goes back out: the set it would
            # make is unsolvable whatever leaves.
            if _active_set_supports_a_solution(prob, active, act_ph)
                last_added = i_best
            else
                j = findfirst(==(i_best), active)
                if j !== nothing
                    active = active[setdiff(eachindex(active), [j])]
                    xB = xB[setdiff(eachindex(xB), [j])]
                end
                push!(rejected, i_best)
            end
        end
        if !isempty(cand_ph)
            k_best = let q = q
                cand_ph[argmax([_admission_measure!(admission, prob, k, u, x_buf, q, dead) for k in cand_ph])]
            end
            push!(act_ph, k_best)
            push!(refs, PHASE_ADMISSION_SEED)
            # A mole-fraction phase also spends one unit of stationarity capacity.
            if !_active_set_supports_a_solution(prob, active, act_ph)
                active, xB = _exchange_incumbent(prob, active, xB, act_ph, eachindex(active))
            end
            if !_active_set_supports_a_solution(prob, active, act_ph)
                pop!(act_ph)
                pop!(refs)
            end
        end

        drop_ph_prev = drop_ph
        key = (sort(copy(act_ph)), sort(copy(active)))
        key in seen && break
        push!(seen, key)
    end

    # Return the best state the search passed through, not the last one tried.
    #
    # `v` already holds the solution of the inner Newton on that active set, so
    # there is nothing to re-solve: refs, `y` and the bounded amounts are read back
    # out of it and the residual is evaluated once to set `converged`.
    #
    # `converged` needs the state's admission violation as well as its residual.
    # The residual is that of the subproblem the active set defines, and a set
    # that holds a supersaturated phase out solves its own equations exactly.
    # Judged on its residual alone, such a state was measured with
    # `converged = true` next to a `kkt_error` of 9, which the certificate
    # refuses; and the multi-start loop in `dual_newton_solve` stops at the
    # first converged attempt, so the flag would end the search on a point that
    # is not a solution.
    if best_state !== nothing
        act_ph, active, v, W, best_viol = best_state
        nph = length(act_ph)
        na = length(active)
        refs = let v = v
            [exp(v[a]) for a in 1:nph]
        end
        y = v[(nph + 1):(nph + m)]
        xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]
        q = nq == 0 ? Float64[] : v[(nph + m + na + 1):(nph + m + na + nq)]
        converged = _judged_residual(
            _outer_residual(
                prob, v, W, act_ph, active, bv, x_buf;
                dead, degenerate, inner_maxit = opts.inner_maxit, max_fall = opts.inner_fall_bound,
            ),
            _judged_scales(_balance_scales(prob, x_buf, q, bv, degenerate), bv), nph,
        ) <= opts.tol && best_viol <= opts.si_tol
    end

    _invert_phases!(
        prob, W, y, refs, act_ph, active, xB, x_buf;
        dead = dead, g = current_g(prob, q), q = q, maxsweeps = opts.inner_maxit,
        max_fall = opts.inner_fall_bound,
    )
    x = copy(x_buf)
    for (j, i) in enumerate(active)
        x[i] = max(xB[j], 0.0)
    end

    return (;
        x = x, y = y, q = q, active_phases = act_ph, active = active,
        converged = converged, kkt_error = best_res,
    )
end

"""
    kkt_certificate(prob, x, b; floor = 1e-25, tol = 1e-10, si_tol = 1e-8, q = prob.q0)
        -> NamedTuple

Check the KKT conditions at `x`, independently of how it was obtained. For a
convex problem they are sufficient, so `optimal = true` is a **proof**.

`tol` is the threshold of the stationarity and balance tests, `si_tol` that of the
saturation and tangent-plane tests, and `q` the unknown parameters `x` was solved
with, when the problem has some.

The result holds `optimal` and what it was decided on: `stationarity` (scaled;
`stationarity_abs` unscaled, `stationarity_scale` the scale),
`stationarity_floored` and `n_floored`, `feasibility` (`feasibility_abs`,
`feasibility_rel`), `worst_violation` (the largest of `worst_violation_bounded`,
`worst_violation_phase` and `worst_violation_split`), `absent_phases`,
`split_phases` and `split_trials` (the trial composition of each phase that
wants to split), `n_interior`, `n_forced_zero` and `param_residual`.

# What is checked, and on which variables

A variable is INTERIOR when `xᵢ > floor`. There the condition is the equality
`∇fᵢ + (Aᵀy)ᵢ = 0`, and `y` is obtained from those variables by least squares.
Below `floor` a variable is at its bound, where the condition is the INEQUALITY
`∇fᵢ + (Aᵀy)ᵢ ≥ 0`:

  - for a pure phase and a bounded member, as a saturation index
    (`worst_violation_bounded`, judged against `si_tol`);
  - for a member of a present phase whose potential the interior determines, as
    the one-sided form of the equality, scaled like it (`stationarity_floored`,
    judged against `tol`). Its amount is taken to be the truncation of a smaller
    exact one; a member held below the amount the multipliers give it fails. A
    member whose potential the interior leaves free is not tested: some
    multiplier of the answer meets its inequality;
  - the members of an absent phase are tested together, by the tangent plane
    (`worst_violation_phase`).

Getting that split wrong is not a detail: imposing the equality on a variable
held at `1e-16` whose stationarity value is `e⁻³⁰⁰` misstates `hᵢ` by 263 units,
and the check then reports a residual of 74 for a point solved to `5e-12`.

Variables carrying a component whose right-hand side has vanished are excluded
from both tests: they are zero by the CONSTRAINT, and the multiplier of a
component nobody supplies is determined by nothing.

# The balance

Each row of `A x = b` that has a budget is judged against what it holds,
`Σⱼ |Aₖⱼ xⱼ|`, and in moles once that exceeds a mole; a row whose budget is zero
within rounding, and a degenerate one, in moles. `feasibility` is the worst
`|rₖ| / min(scaleₖ, 1)`, the larger of `feasibility_abs` (moles) and
`feasibility_rel` (relative, over the rows with a budget), and `optimal` asks it
below `tol`. A trace is held to its own amount, and a large row to the tolerance
in moles it always had.
"""
function kkt_certificate(
        prob::DualNewtonProblem, x::AbstractVector, b::AbstractVector;
        floor::Float64 = 1.0e-25, tol::Float64 = 1.0e-10, si_tol::Float64 = 1.0e-8,
        q = prob.q0,
    )
    # A certificate is a verdict on values: a problem, an answer or a budget
    # carrying dual numbers is judged on its primal values, every level down.
    prob = _full_primal_problem(prob)
    xv = Float64[_primal_value(v) for v in x]
    bv = Float64[_primal_value(v) for v in b]
    q = Float64[_primal_value(v) for v in q]
    gq = current_g(prob, q)
    ∇f = _primal_value.(gq .+ current_h(prob, xv, q))

    degenerate = _degenerate_conservation_rows(prob, bv)
    dead = _dead_variables(prob.A, degenerate)

    # WHICH TEST APPLIES TO WHICH VARIABLE.
    #
    # A member of a mixing phase is NEVER at its bound. Its `hᵢ` is a logarithm of
    # a fraction, so it diverges to `−∞` as the amount goes to zero: `xᵢ = 0` is
    # not attainable, and at the solution its condition is the EQUALITY
    # `∇fᵢ + (Aᵀy)ᵢ = 0`, however small it is. Only a bound-constrained variable —
    # a pure phase, `hᵢ ≡ 0` — can sit exactly at zero and obey the inequality.
    #
    # Confusing the two is not a detail. Applying the inequality `uᵢ ≤ ∇fᵢ` to a
    # phase member truncated at the numerical floor reads `uᵢ ≤ gᵢ − 700`, which no
    # finite multiplier satisfies, and the certificate then reports a
    # supersaturation of several hundred for a composition that is optimal. On a
    # cement without limestone that alone lost four of forty instants.
    #
    # A phase member BELOW the floor is excluded from both tests: it is not a
    # statement about the chemistry but the truncation of `exp(w)` at `w = −700`,
    # and its exact value would satisfy the equality.
    in_phase = Set{Int}()
    for ph in prob.phases, i in ph.members
        push!(in_phase, i)
    end

    # The exception is a member declared bounded (`SolutionPhase`): its activity
    # stays finite as it vanishes, so it CAN be exactly absent from a present
    # phase, and below the floor it is tested by the inequality a pure phase
    # obeys. Left excluded, a gel missing a member it should hold would pass.
    bounded = Set{Int}()
    for ph in prob.phases
        any(xv[i] > floor for i in ph.members) || continue
        for j in ph.bounded_members
            push!(bounded, ph.members[j])
        end
    end

    interior = [i for i in eachindex(xv) if xv[i] > floor && !(i in dead)]
    at_bound = [
        i for i in eachindex(xv)
            if xv[i] <= floor && !(i in dead) && (!(i in in_phase) || i in bounded)
    ]

    Ai = Matrix(@view prob.A[:, interior])
    # Pivoted QR, not `\`: when the interior count equals the number of rows the
    # matrix is square and Julia reaches for LU, which throws on a rank-deficient
    # set — and rank deficiency is ordinary here.
    y = qr(transpose(Ai), ColumnNorm()) \ (-∇f[interior])

    # SCALED, as Wächter & Biegler (2006) Eq. (5) scales Ipopt's `E_0`. The
    # entries of `∇f` are chemical potentials referred to the elements, of order
    # 10²-10³ in RT units, so an absolute threshold of 1e-10 on their residual
    # demands thirteen digits of cancellation. On a system whose multipliers are
    # that large — a kinetic step pinning a mineral by a linear row, whose
    # multiplier must reach the mineral's own potential — the answer came out
    # right to nine digits while the certificate reported 2.9e-8 and refused it.
    #
    # The divisor is the size of the quantities the residual is built from, never
    # below one, so a well-scaled problem is judged as it would be without it.
    stat_raw = isempty(interior) ? 0.0 :
        maximum(abs, ∇f[interior] .+ transpose(Ai) * y)
    stat_scale = max(
        1.0,
        isempty(interior) ? 0.0 : maximum(abs, @view ∇f[interior]),
        isempty(y) ? 0.0 : maximum(abs, y),
    )
    stationarity = stat_raw / stat_scale
    # The linear rows are `A x + Aq q − b`, and forgetting `Aq q` reports the
    # parameter itself as an infeasibility: on a kinetic step the residual came
    # out at exactly `Δξ`.
    resid = prob.nq == 0 ? (prob.A * xv .- bv) :
        (prob.A * xv .+ prob.Aq * collect(q) .- bv)
    # Each row judged against what it holds, in moles above one mole: the
    # measure the solver converges on (`_balance_scales`). Judged in moles alone,
    # a trace of 1e-9 mol could be 10 % wrong and certified.
    #
    # A relative measure might be feared to refuse correct answers: the charge
    # row of a dilute solution, `b = 0` and a flux of 1e-6, would turn 7.6e-7
    # mol of machine noise into 0.76. Measured on a dilute
    # sodium chloride, the dual Newton leaves 3e-21 mol on a 1e-6 mol row,
    # 3e-15 of it; a residual of 76 % of a row's flux is no rounding. The rows
    # whose relative residual is large at a correct answer are those of a
    # component nobody supplies, its carriers at the floor: degenerate rows,
    # judged in moles.
    scales = _balance_scales(prob, xv, q, bv, degenerate)
    bscale = max(1.0, maximum(abs, bv; init = 0.0))
    feas_abs = maximum(abs, resid; init = 0.0)
    feas_rel = 0.0
    for k in eachindex(resid)
        # Relative only where the row has a total: see `_judged_scales`.
        (k in degenerate || abs(bv[k]) <= eps() * bscale) && continue
        feas_rel = max(feas_rel, abs(resid[k]) / scales[k])
    end
    feasibility = max(feas_abs, feas_rel)

    # The NONLINEAR residual of the parameter block. Leaving it out was a hole,
    # not an omission of detail: a kinetic step whose mineral is dropped from the
    # active set satisfies its reactivity row trivially — `Δξ` simply takes the
    # whole amount — and satisfies stationarity and the element balance too, so it
    # certified while violating the one equation that makes it a KINETIC step,
    # `Δξ − Δt·M·r(n) = 0`. Measured, a march that should have stopped at
    # saturation dissolved everything and was proved optimal.
    param_residual = prob.nq == 0 ? 0.0 :
        _primal_value(maximum(abs, prob.cq(xv, collect(q), prob.params)))

    u = -(transpose(prob.A) * y)
    worst = isempty(at_bound) ? -Inf : maximum(u[i] - ∇f[i] for i in at_bound)

    # A member of a PRESENT phase below the floor is excluded from the equality,
    # on the ground that its amount is the truncation of `exp(w)`: the exact
    # amount is smaller still, so the member would give matter back rather than
    # take it, `uᵢ − ∇fᵢ ≤ 0`. That is the one test it must pass, and it was not
    # made. A member held far BELOW its equilibrium amount wants the reverse, and
    # went unexamined: measured on a cement, H+ at 3e-100 mol in a solution whose
    # potentials give it 1.2e-16, certified, and a pH read from that amount came
    # out 0.09 high. Bounded members are tested with the pure phases above, and
    # the members of an absent phase by the tangent plane below.
    #
    # Only where the interior DETERMINES the member's potential. `y` is fixed by
    # the interior up to the null space of `Aiᵀ`, and the least-squares `y` above
    # is the one of minimum norm in it; a member whose column has a component in
    # that null space has a potential the interior leaves free, and its
    # inequality can be met by some multiplier of the answer whatever this one
    # says. Measured: on a calcite solution the redox direction is free (no
    # interior species carries it), the minimum-norm multiplier made H2⁰ at
    # 1e-305 mol "want" to rise by 0.07, and the multipliers of the solve itself
    # hold it 17 units below. That was a refusal of a correct answer.
    free_directions = isempty(interior) ? Matrix{Float64}(I, size(prob.A, 1), size(prob.A, 1)) :
        nullspace(transpose(Ai))
    determined(i) = begin
        a = @view prob.A[:, i]
        isempty(free_directions) || norm(transpose(free_directions) * a) <= 1.0e-8 * max(1.0, norm(a))
    end
    floored = Int[]
    for ph in prob.phases
        any(xv[i] > floor && !(i in dead) for i in ph.members) || continue
        for i in ph.members
            xv[i] <= floor && !(i in dead) && !(i in bounded) && determined(i) && push!(floored, i)
        end
    end
    # The one-sided form of the SAME condition the interior obeys, so it is
    # scaled and judged as `stationarity` is: a member at its exact amount below
    # the floor satisfies the equality to the same rounding, and an absolute
    # threshold would refuse it on noise.
    stationarity_floored = isempty(floored) ? 0.0 :
        max(0.0, maximum(u[i] - ∇f[i] for i in floored)) / stat_scale

    # A mixing phase held ENTIRELY absent is tested by neither of the two above:
    # its members are excluded from `interior` (they are at the floor) and from
    # `at_bound` (they are phase members, whose condition is an equality). So a
    # composition that omits a solid solution which should have formed passed the
    # certificate unexamined — the one soundness hole the certificate had, and
    # exactly the case that matters for a cement, where the C-S-H is a mixing
    # phase. Michelsen's measure is the test such a phase requires.
    worst_phase = -Inf
    absent_phases = Int[]
    for (k, ph) in pairs(prob.phases)
        all(xv[i] <= floor || i in dead for i in ph.members) || continue
        all(i in dead for i in ph.members) && continue   # cannot exist at all
        push!(absent_phases, k)
        worst_phase = max(
            worst_phase,
            _primal_value(phase_tangent_measure(prob, k, u, xv; g = gq, q = q, dead = dead)),
        )
    end
    # A mixing phase that is PRESENT is tested by the stationarity of its members
    # and by nothing else, and stationarity is blind to the one failure that
    # matters for a non-ideal phase: that the Gibbs minimum for it is TWO
    # coexisting compositions rather than the one reported. Convexity is what
    # made that safe to ignore, and a Redlich-Kister excess term strong enough to
    # open a miscibility gap is exactly the case where convexity fails.
    #
    # Costs nothing on a convex system: at equilibrium the measure below is zero
    # by construction there, so it cannot move `worst_all`.
    worst_split = -Inf
    split_phases = Int[]
    # The composition each flagged phase wants to split INTO, as mole fractions
    # over that phase's members. Michelsen's stability analysis produces it as a
    # by-product of the test, and a caller that means to act on the verdict --
    # by giving the phase a second instance and starting it there -- cannot get
    # it any other way: the pair is a property of the FULL system, fixed jointly
    # with the solution the phase sits in, and not of the mixing model alone.
    split_trials = Dict{Int, NamedTuple{(:members, :x), Tuple{Vector{Int}, Vector{Float64}}}}()
    for (k, ph) in pairs(prob.phases)
        length(ph.members) < 2 && continue
        # Mole-fraction phases only. The aqueous solution is a phase here too,
        # but its activities are molalities referred to the solvent, not mole
        # fractions of its own members, and the tangent-plane measure is written
        # in the latter -- applied to it, it measures nothing meaningful. It also
        # cannot unmix: there is one solvent.
        ph.mole_fraction || continue
        ph.always_present && continue
        any(xv[i] > floor && !(i in dead) for i in ph.members) || continue
        m, trial = phase_split_trial(prob, k, u, xv; g = gq, q = q, dead = dead)
        m, trial = _primal_value(m), _primal_value.(trial)
        if m > worst_split
            worst_split = m
        end
        # Flagged on the SAME tolerance the verdict uses. At a converged
        # equilibrium the measure is zero up to rounding, and `m > 0` alone
        # would name a phase on 1e-12 of numerical noise.
        if m > si_tol
            push!(split_phases, k)
            # Keyed by the phase's position in `prob.phases`, and carrying the
            # indices of its members, so that a caller needs no table mapping
            # the solver's phase list back to its own species.
            isempty(trial) ||
                (split_trials[k] = (members = collect(ph.members), x = trial))
        end
    end

    worst_all = max(worst, worst_phase, worst_split)

    return (;
        stationarity = stationarity, stationarity_abs = stat_raw,
        stationarity_scale = stat_scale, feasibility = feasibility,
        feasibility_abs = feas_abs, feasibility_rel = feas_rel,
        worst_violation = worst_all, worst_violation_bounded = worst,
        stationarity_floored = stationarity_floored, n_floored = length(floored),
        worst_violation_phase = worst_phase, absent_phases = absent_phases,
        worst_violation_split = worst_split, split_phases = split_phases,
        split_trials = split_trials,
        n_interior = length(interior),
        n_forced_zero = length(dead),
        param_residual = param_residual,
        optimal = stationarity <= tol && stationarity_floored <= tol && feasibility <= tol &&
            worst_all <= si_tol && param_residual <= max(tol, si_tol),
    )
end
