using LinearAlgebra

@testset "degenerate components are recognised by sign, not by zero" begin

    # With `x ≥ 0`, the row `Σᵢ A_{ki} xᵢ = b_k` forces every `xᵢ` with
    # `A_{ki} ≠ 0` to vanish when `b_k = 0` ONLY IF the non-zero entries of the
    # row share a sign: a sum of non-negative terms vanishes only term by term. A
    # row with entries of both signs permits cancellation and forces nothing.

    # All entries positive, zero right-hand side: degenerate.
    @test degenerate_components(Float64[1 2 3], [0.0]) == [1]

    # Same row, non-zero right-hand side: not degenerate.
    @test isempty(degenerate_components(Float64[1 2 3], [1.0]))

    # All entries negative, zero right-hand side: degenerate too — the sign has
    # to be constant, not positive.
    @test degenerate_components(Float64[-1 -2], [0.0]) == [1]

    # Mixed signs and zero right-hand side: NOT degenerate. In a chemical system
    # this is the acid–base row, where `H+` carries `+1` and `OH-` carries `−1`,
    # and treating it as degenerate removes the entire aqueous system.
    @test isempty(degenerate_components(Float64[1 -1 0], [0.0]))

    # Several rows at once, with the threshold relative to the largest entry of
    # `b`: a component 12 orders below the scale counts as vanished.
    A = Float64[1 1 0; 0 0 1; 1 -1 0]
    @test degenerate_components(A, [50.0, 0.0, 0.0]) == [2]
    @test degenerate_components(A, [50.0, 1.0e-14, 0.0]) == [2]
    @test isempty(degenerate_components(A, [50.0, 1.0, 0.0]))

end

@testset "dual_newton_solve on a problem with a known answer" begin

    # A two-variable ideal mixture with one linear constraint, whose minimizer is
    # available in closed form. With `f = Σ xᵢ(gᵢ + ln xᵢ)` and `x₁ + x₂ = 1`,
    # stationarity gives `g₁ + ln x₁ = g₂ + ln x₂`, so
    # `x₁ = 1/(1 + e^{g₁-g₂})`.
    g = [0.0, 1.0]
    h(x, _) = log.(max.(x, 1.0e-300))
    A = Float64[1 1]
    b = [1.0]

    prob = DualNewtonProblem(
        A, g, h;
        phases = [SolutionPhase([1, 2], 2; always_present = true)],
        idx_bounded = Int[],
    )
    res = dual_newton_solve(prob, b, [0.5, 0.5])

    x1 = 1 / (1 + exp(g[1] - g[2]))
    @test res.x[1] ≈ x1 rtol = 1.0e-8
    @test res.x[2] ≈ 1 - x1 rtol = 1.0e-8
    @test res.converged

    cert = kkt_certificate(prob, res.x, b)
    @test cert.optimal
    @test cert.stationarity < 1.0e-9
    @test cert.feasibility < 1.0e-9

    # And the certificate must be able to say NO: perturb the answer and it does.
    # The perturbation must stay in the domain: `x₁ ≈ 0.731`, so a factor 1.2
    # leaves `x₂ ≈ 0.123 > 0`. A factor 1.5 would send `x₂` negative and the
    # logarithm undefined, which tests nothing.
    bad = [1.2 * res.x[1], 1 - 1.2 * res.x[1]]
    @test all(>(0), bad)
    @test !kkt_certificate(prob, bad, b).optimal

end

@testset "a variable cannot be in two places at once" begin

    h(x, _) = log.(max.(x, 1.0e-300))
    # A variable in two phases, and a variable both in a phase and bounded, are
    # both contradictions: the first has two mixing contexts, the second is asked
    # to be interior and to be allowed to vanish.
    @test_throws ArgumentError DualNewtonProblem(
        Float64[1 1 1], zeros(3), h;
        phases = [SolutionPhase([1, 2], 1), SolutionPhase([2, 3], 1)],
    )
    @test_throws ArgumentError DualNewtonProblem(
        Float64[1 1 1], zeros(3), h;
        phases = [SolutionPhase([1, 2], 1)], idx_bounded = [2],
    )
    # And `j_ref` must point inside the phase.
    @test_throws ArgumentError DualNewtonProblem(
        Float64[1 1], zeros(2), h; phases = [SolutionPhase([1, 2], 3)],
    )

end

@testset "an empty set of mixing phases is refused" begin

    # The multipliers are determined by the stationarity of the strictly positive
    # variables. With none, nothing determines them, and saying so beats
    # returning a number.
    @test_throws ArgumentError DualNewtonProblem(
        Float64[1 1], [0.0, 1.0], (x, _) -> log.(x);
        phases = SolutionPhase[], idx_bounded = [1, 2],
    )

end

@testset "a mixing phase is admitted by the tangent plane, not by a sign" begin

    # Two variables that mix ideally, and one bound-constrained variable that
    # does not. The mixing pair obeys `hᵢ = ln(xᵢ/N)`; the third has `h ≡ 0` and
    # is either present at its stationarity or exactly zero.
    #
    # The distinction the solver has to make: a member of the mixing phase is
    # never exactly absent while the phase exists, so no saturation index decides
    # it — only the phase as a whole is admitted or not, on Michelsen's measure
    # `Σᵢ exp(uᵢ − gᵢ) > 1`.
    g = [0.0, 1.0, 5.0]
    function h(x, _)
        N = x[1] + x[2]
        return [
            log(max(x[1], 1.0e-300) / max(N, 1.0e-300)),
            log(max(x[2], 1.0e-300) / max(N, 1.0e-300)), 0.0,
        ]
    end
    A = Float64[1 1 1]
    b = [1.0]

    prob = DualNewtonProblem(
        A, g, h;
        phases = [SolutionPhase([1, 2], 2; always_present = true)],
        idx_bounded = [3],
    )
    res = dual_newton_solve(prob, b, [0.5, 0.5, 1.0e-9])
    cert = kkt_certificate(prob, res.x, b)

    @test cert.optimal
    @test cert.feasibility < 1.0e-9

    # `g₃ = 5` is far above the potential the mixture settles at, so the
    # bound-constrained variable must stay out — and it is the ONLY variable that
    # can be exactly zero.
    @test res.x[3] < 1.0e-8
    @test res.x[1] > 0 && res.x[2] > 0

    # The mixture itself follows the ideal law `x₁/x₂ = exp(g₂ − g₁)`.
    @test res.x[1] / res.x[2] ≈ exp(g[2] - g[1]) rtol = 1.0e-6

end

@testset "a second mixing phase is admitted only when it is stable" begin

    # Two mixing phases sharing one component: the first always present, the
    # second — a solid solution — admitted or not by the tangent-plane measure
    # `Σᵢ exp(uᵢ − gᵢ) > 1` over ITS members. A saturation index cannot decide
    # this: the phase has no single one, and its members are never exactly absent
    # while it exists.
    function h2(x, _)
        NA = x[1] + x[2]
        NB = x[3] + x[4]
        return [
            log(max(x[1], 1.0e-300) / max(NA, 1.0e-300)),
            log(max(x[2], 1.0e-300) / max(NA, 1.0e-300)),
            log(max(x[3], 1.0e-300) / max(NB, 1.0e-300)),
            log(max(x[4], 1.0e-300) / max(NB, 1.0e-300)),
        ]
    end
    A2 = Float64[1 1 1 1]
    b2 = [1.0]
    phases(alwaysB) = [
        SolutionPhase([1, 2], 2; always_present = true),
        SolutionPhase([3, 4], 1; always_present = alwaysB),
    ]

    # The first phase settles at `u = −ln(e^{-g₁} + e^{-g₂}) = −ln(1 + e^{-1})`.
    u_A = -log(1 + exp(-1.0))

    # Members far ABOVE that potential: the second phase cannot form.
    prob_out = DualNewtonProblem(
        A2, [0.0, 1.0, 8.0, 9.0], h2; phases = phases(false), idx_bounded = Int[],
    )
    r_out = dual_newton_solve(prob_out, b2, [0.5, 0.5, 1.0e-9, 1.0e-9])
    @test kkt_certificate(prob_out, r_out.x, b2).optimal
    @test r_out.x[3] + r_out.x[4] < 1.0e-6
    @test r_out.x[1] / r_out.x[2] ≈ exp(1.0) rtol = 1.0e-6

    # Members far BELOW it: the tangent-plane measure is positive and the phase
    # must be ADMITTED. What the two phases then settle at is a different
    # question — with a single component the phase rule leaves them no room to
    # coexist, so one of them must empty — and the assertion here is about the
    # admission, which is what the criterion decides.
    prob_in = DualNewtonProblem(
        A2, [0.0, 1.0, -8.0, -9.0], h2; phases = phases(false), idx_bounded = Int[],
    )
    r_in = dual_newton_solve(prob_in, b2, [0.5, 0.5, 1.0e-9, 1.0e-9])
    @test 2 in r_in.active_phases
    @test r_in.x[3] + r_in.x[4] > r_out.x[3] + r_out.x[4]
    @test u_A < 0                       # the reference potential, for the record

end

@testset "an active set has to satisfy the phase rule" begin

    # A bound-constrained variable held active imposes `aᵢᵀy = −gᵢ`, one linear
    # equation in `y`; a mole-fraction phase imposes `logsumexp(uᵢ − gᵢ) = 0`, one
    # more. With `y ∈ ℝᵐ` there is no `y` satisfying more than `m` of them, so an
    # active set carrying more cannot support a solution at any iterate — the
    # least-squares step spreads the violation instead of removing it.
    #
    # Four pure phases in a two-component system: at most two can coexist, and the
    # solver must return an assemblage that respects it rather than grinding on an
    # unsolvable one.
    A = Float64[1 0 1 1 2; 0 1 1 2 1]
    g = [0.0, 0.0, -1.0, -2.0, -1.5]
    h(x, _) = zeros(length(x))          # every variable a pure phase
    prob = DualNewtonProblem(
        A, g, h;
        phases = [SolutionPhase([1, 2], 1; always_present = true)],
        idx_bounded = [3, 4, 5],
    )
    @test stationarity_capacity(prob) == 2

    b = [1.0, 1.0]
    r = dual_newton_solve(prob, b, [0.5, 0.5, 0.1, 0.1, 0.1])

    # Whatever it returns, it must not hold more phases stationary than the
    # components allow: three bound variables plus the solution's own condition
    # would be four conditions on two multipliers.
    n_cond = length(r.active) + count(k -> prob.phases[k].mole_fraction, r.active_phases)
    @test n_cond <= stationarity_capacity(prob)

    # And the active bounded variables must be linearly independent, else their
    # `uᵢ = gᵢ` conditions fix a relation between the `gᵢ` that does not hold.
    if !isempty(r.active)
        @test rank(A[:, r.active]) == length(r.active)
    end

    # Matter is conserved whatever the active set.
    @test A * r.x ≈ b rtol = 1.0e-6

end

@testset "the certificate tests a mixing phase held absent" begin

    # A mixing phase whose members are ALL at the floor is examined by neither of
    # the certificate's two tests: its members are not interior (they sit at the
    # floor) and not bound-constrained (a phase member's condition is an equality,
    # since `ln xᵢ → −∞`). So a composition that omits a solid solution which
    # should have formed used to pass unexamined. That is what
    # `phase_tangent_measure` closes.
    #
    # Setup: an always-present carrier phase (variable 1), a two-member ideal
    # mixing phase (2, 3) and one conservation row per component.
    #
    #   component 1: carried by variable 1 and by member 2
    #   component 2: carried by member 3 only
    #
    # With component 2 present in the budget, the mixing phase MUST form: nothing
    # else can hold it.
    A = Float64[1 1 0; 0 0 1]
    g = [0.0, 0.0, 0.0]
    h(x, _) = begin
        N = max(x[2] + x[3], 1.0e-300)
        [log(max(x[1], 1.0e-300)),
         log(max(x[2], 1.0e-300) / N),
         log(max(x[3], 1.0e-300) / N)]
    end
    prob = DualNewtonProblem(
        A, g, h;
        phases = [
            SolutionPhase([1], 1; always_present = true),
            SolutionPhase([2, 3], 1; mole_fraction = true),
        ],
        idx_bounded = Int[],
    )
    b = [1.0, 0.2]

    # An honest solve forms the phase and certifies.
    res = dual_newton_solve(prob, b, [0.9, 0.1, 0.2])
    cert_ok = kkt_certificate(prob, res.x, b)
    @test res.x[2] + res.x[3] > 1.0e-6
    @test isempty(cert_ok.absent_phases)

    # Now hand the certificate a composition with the mixing phase suppressed.
    # It cannot satisfy the second conservation row, so build the test on the
    # measure itself: at these multipliers the phase is stable and the measure
    # must be positive.
    x_absent = [1.0, 0.0, 0.0]
    u = -(transpose(A) * [0.0, -5.0])       # a potential that favors member 3
    @test phase_tangent_measure(prob, 2, u, x_absent) > 0

    # And a potential that does not. BOTH members must be far below the plane,
    # not just one: with `y = [0, 50]` member 2 still has `u₂ − g₂ = 0`, so it
    # sits exactly at its stability limit and the measure is 0.0 — which is the
    # right answer, and the reason this test names both multipliers.
    @test phase_tangent_measure(prob, 2, -(transpose(A) * [0.0, 50.0]), x_absent) == 0.0
    u_low = -(transpose(A) * [50.0, 50.0])
    @test phase_tangent_measure(prob, 2, u_low, x_absent) < -40

    # The measure is reported and enters `optimal`.
    cert = kkt_certificate(prob, x_absent, [1.0, 0.0])
    @test 2 in cert.absent_phases
    @test isfinite(cert.worst_violation_phase) || cert.worst_violation_phase == -Inf
    @test cert.worst_violation == max(cert.worst_violation_bounded,
                                      cert.worst_violation_phase)

    # For an ideal phase the measure does not depend on the trial amount.
    m1 = phase_tangent_measure(prob, 2, u, x_absent; total = 1.0e-6)
    m2 = phase_tangent_measure(prob, 2, u, x_absent; total = 1.0e-3)
    @test isapprox(m1, m2; atol = 1.0e-10)

end

@testset "the q block solves for a prescribed property" begin

    # Two ideal species in one phase, `x₁ + x₂ = 1`, whose standard potentials
    # depend on an unknown parameter `q`:
    #
    #     g(q) = [0, q]
    #
    # With `f = Σ xᵢ(gᵢ(q) + ln xᵢ)` the minimizer at fixed `q` is
    # `x₁ = 1/(1 + e^{-q})`. Prescribing `x₁ = target` therefore FIXES `q`, and
    # the closed form inverts: `q = ln(target/(1 − target))`.
    #
    # That is the same structure as an adiabatic solve — a property prescribed,
    # a parameter unknown, both in one square system — with an answer available
    # in closed form, which an enthalpy balance does not have.
    target = 0.6
    A = Float64[1 1]
    b = [1.0]
    h(x, _) = log.(max.(x, 1.0e-300))

    prob = DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 2; always_present = true)],
        idx_bounded = Int[],
        gq = (q, _) -> [0.0, q[1]],
        cq = (x, q, _) -> [x[1] - target],
        q0 = [0.1],
        qscale = [1.0],
    )
    res = dual_newton_solve(prob, b, [0.5, 0.5])

    @test res.converged
    @test res.x[1] ≈ target rtol = 1.0e-8
    @test res.x[2] ≈ 1 - target rtol = 1.0e-8
    @test res.q[1] ≈ log(target / (1 - target)) rtol = 1.0e-6

    # And the certificate must be evaluated at the parameters that were found,
    # not at the starting guess: `∇f` depends on `q`.
    c_found = kkt_certificate(prob, res.x, b; q = res.q)
    @test c_found.stationarity < 1.0e-8
    c_guess = kkt_certificate(prob, res.x, b; q = prob.q0)
    @test c_guess.stationarity > 1.0e-3

    # A `qscale` of the wrong length is refused, because it sets the difference
    # step and a silent default would secant across the wrong interval.
    @test_throws ArgumentError DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 2; always_present = true)],
        gq = (q, _) -> [0.0, q[1]], cq = (x, q, _) -> [x[1] - target],
        q0 = [0.1], qscale = Float64[],
    )
    # And `q0` without the two callbacks is refused too.
    @test_throws ArgumentError DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 2; always_present = true)],
        q0 = [0.1], qscale = [1.0],
    )

end

@testset "hq lets the activity model see the unknown parameters" begin

    # `h` depending on `q` is not a corner case: the Debye-Hückel coefficients are
    # functions of temperature, so an adiabatic solve whose activity model stayed
    # at the starting temperature would minimize the wrong Gibbs energy.
    #
    # Here `h` carries an ideal `ln x` plus a term `q·x₁`, a regular-solution
    # excess whose strength IS the unknown. Prescribing `x₁` fixes `q`, and the
    # stationarity `g₁ + ln x₁ + q x₁ = g₂ + ln x₂` inverts in closed form:
    #
    #     q = (ln x₂ − ln x₁) / x₁    with g = 0
    target = 0.6
    A = Float64[1 1]
    b = [1.0]
    h_fixed(x, _) = log.(max.(x, 1.0e-300))            # never used: hq wins
    hq(x, q, _) = [log(max(x[1], 1.0e-300)) + q[1] * x[1],
                   log(max(x[2], 1.0e-300))]

    prob = DualNewtonProblem(
        A, [0.0, 0.0], h_fixed;
        phases = [SolutionPhase([1, 2], 2; always_present = true)],
        gq = (q, _) -> [0.0, 0.0],
        hq = hq,
        cq = (x, q, _) -> [x[1] - target],
        q0 = [0.0], qscale = [1.0],
    )
    res = dual_newton_solve(prob, b, [0.5, 0.5])

    @test res.converged
    @test res.x[1] ≈ target rtol = 1.0e-8
    @test res.q[1] ≈ (log(1 - target) - log(target)) / target rtol = 1.0e-6

    # Without `hq` the same problem cannot reach the target, because nothing in
    # the residual then depends on `q` except through `gq`, which is constant.
    prob_no_hq = DualNewtonProblem(
        A, [0.0, 0.0], h_fixed;
        phases = [SolutionPhase([1, 2], 2; always_present = true)],
        gq = (q, _) -> [0.0, 0.0],
        cq = (x, q, _) -> [x[1] - target],
        q0 = [0.0], qscale = [1.0],
    )
    res_no = dual_newton_solve(prob_no_hq, b, [0.5, 0.5])
    @test !res_no.converged

end

@testset "Aq puts a reaction extent in the linear rows" begin

    # The structure of a kinetic step (Leal et al. 2017), on a problem small
    # enough to solve by hand.
    #
    # Two species in one ideal phase, one reaction `1 → 2`, so `K = [-1, 1]`.
    # The linear rows are
    #
    #     x₁ + x₂           = 1        (the element balance)
    #     Kᵀx      − Δξ     = ξ₀       (the reactivity constraint)
    #
    # and the nonlinear one is `Δξ − Δt·M·r = 0` with `M = KᵀK = 2`. With a
    # CONSTANT rate the answer is closed form: `Δξ = 2·Δt·r`, and the composition
    # follows from the two linear rows.
    Δt, rate = 0.5, 0.3
    K = reshape([-1.0, 1.0], 2, 1)
    M = (transpose(K) * K)[1, 1]                # = 2
    A = Float64[1 1; -1 1]                      # element row, then Kᵀ
    Aq = reshape([0.0, -1.0], 2, 1)             # −Δξ in the second row only
    ξ0 = -1.0 + 2 * 0.1                         # start at x = [0.9, 0.1]
    b = [1.0, ξ0]

    h(x, _) = log.(max.(x, 1.0e-300))
    prob = DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 1; always_present = true)],
        gq = (q, _) -> [0.0, 0.0],
        cq = (x, q, _) -> [q[1] - Δt * M * rate],
        Aq = Aq,
        q0 = [0.0], qscale = [1.0],
    )
    res = dual_newton_solve(prob, b, [0.9, 0.1])

    Δξ = Δt * M * rate                          # 0.3
    @test res.converged
    @test res.q[1] ≈ Δξ rtol = 1.0e-8
    # The two linear rows are satisfied exactly, extent included.
    @test res.x[1] + res.x[2] ≈ 1.0 atol = 1.0e-12
    @test -res.x[1] + res.x[2] - res.q[1] ≈ ξ0 atol = 1.0e-10
    # And the composition is the one those rows force: x₂ − x₁ = ξ₀ + Δξ.
    @test res.x[2] - res.x[1] ≈ ξ0 + Δξ atol = 1.0e-10

    # A rate law reading the composition works the same way, the extent being an
    # unknown of the same system rather than of an outer iteration. First-order in
    # x₁: `r = k x₁`, so `Δξ = Δt·M·k·x₁` at the END of the step — backward Euler.
    k = 0.4
    prob2 = DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 1; always_present = true)],
        gq = (q, _) -> [0.0, 0.0],
        cq = (x, q, _) -> [q[1] - Δt * M * k * x[1]],
        Aq = Aq, q0 = [0.0], qscale = [1.0],
    )
    res2 = dual_newton_solve(prob2, b, [0.9, 0.1])
    @test res2.converged
    @test res2.q[1] ≈ Δt * M * k * res2.x[1] rtol = 1.0e-8   # implicit, not explicit
    @test res2.q[1] > Δt * M * k * 0.9 * 0.5                 # and not the initial value

    # `Aq` of the wrong shape is refused.
    @test_throws ArgumentError DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 1; always_present = true)],
        gq = (q, _) -> [0.0, 0.0], cq = (x, q, _) -> [0.0],
        Aq = zeros(1, 1), q0 = [0.0], qscale = [1.0],
    )

end
