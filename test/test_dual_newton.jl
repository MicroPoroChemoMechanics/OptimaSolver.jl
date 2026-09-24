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
        [
            log(max(x[1], 1.0e-300)),
            log(max(x[2], 1.0e-300) / N),
            log(max(x[3], 1.0e-300) / N),
        ]
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
    @test cert.worst_violation == max(
        cert.worst_violation_bounded,
        cert.worst_violation_phase
    )

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
    hq(x, q, _) = [
        log(max(x[1], 1.0e-300)) + q[1] * x[1],
        log(max(x[2], 1.0e-300)),
    ]

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

@testset "the degeneracy criterion belongs to conservation rows only" begin

    # `degenerate_components` answers a question about element conservation: a row
    # whose non-zero entries share a sign and whose budget is zero forces every
    # variable in it to vanish. Asked of a row that means something else, it gets
    # the wrong answer.
    #
    # The row that provoked this is a reactivity constraint,
    # `nᵢ − Σⱼ νᵢⱼ Δξⱼ = nᵢ(0)`: a single positive entry on `x` and, for a product
    # that starts absent, a zero right-hand side — exactly the shape the criterion
    # reads as "this component is absent from the system". It then pinned that
    # row's multiplier and declared the species dead, so a solid product could
    # never form. Measured downstream, the stationarity residual sat at 458 and
    # the reaction extents came out 4.3 times short.
    #
    # Two variables, one element row, one "pinning" row on variable 2 whose budget
    # is zero because that variable starts absent.
    A = Float64[1 1; 0 1]
    b = [1.0, 0.0]

    # Read as conservation, the second row is degenerate and kills variable 2.
    @test degenerate_components(A, b) == [2]

    h(x, _) = log.(max.(x, 1.0e-300))
    prob_all = DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 1; always_present = true)],
    )
    @test OptimaSolver._degenerate_conservation_rows(prob_all, b) == [2]

    # Told that only the first row conserves anything, nothing is degenerate.
    prob_one = DualNewtonProblem(
        A, [0.0, 0.0], h;
        phases = [SolutionPhase([1, 2], 1; always_present = true)],
        conservation_rows = [1],
    )
    @test isempty(OptimaSolver._degenerate_conservation_rows(prob_one, b))

    # The default is every row, so a system that declares nothing behaves exactly
    # as before.
    @test prob_all.conservation_rows == collect(1:2)

end

@testset "the certificate's stationarity is scaled" begin

    # The entries of `∇f` are chemical potentials referred to the elements, of
    # order 10²-10³ in RT units. An absolute threshold of 1e-10 on their residual
    # asks for thirteen digits of cancellation, which Float64 does not have: on a
    # kinetic step pinning a mineral by a linear row — whose multiplier must reach
    # that mineral's own potential — the answer was right to nine digits while the
    # certificate reported 2.9e-8 and refused it.
    g = [0.0, 1.0]
    h(x, _) = log.(max.(x, 1.0e-300))
    A = Float64[1 1]
    b = [1.0]
    prob = DualNewtonProblem(
        A, g, h; phases = [SolutionPhase([1, 2], 2; always_present = true)],
    )
    res = dual_newton_solve(prob, b, [0.5, 0.5])
    c = kkt_certificate(prob, res.x, b)

    @test c.optimal
    # Both figures are reported, and the scale is at least one so a well-scaled
    # problem is judged exactly as before.
    @test c.stationarity_scale >= 1
    @test c.stationarity ≈ c.stationarity_abs / c.stationarity_scale rtol = 1.0e-12
    @test c.stationarity <= c.stationarity_abs

    # Feasibility stays ABSOLUTE, deliberately: it states how much matter the
    # composition fails to account for, and that is a number of moles. Scaling it
    # row by row was tried and is wrong — the charge row of a dilute solution has
    # a budget of zero and a flux of order 1e-6, so dividing by it turned 7.6e-7
    # mol of machine noise into 0.76 and refused every answer.
    @test c.feasibility == c.feasibility_abs

end

@testset "a present phase that would rather split" begin
    # `phase_tangent_measure` asks whether an ABSENT phase should form.
    # `phase_split_measure` asks whether a PRESENT one should come apart, and
    # the two are different questions: at equilibrium the tangent-plane distance
    # at a present phase's own composition is exactly zero AND is a fixed point
    # of the substitution, so the uniform start used for an absent phase walks
    # straight to it and reports nothing.
    #
    # A symmetric regular solution gives an exact oracle. With
    #
    #     g/RT = x ln x + (1-x) ln(1-x) + A x (1-x)
    #
    # the second derivative at x = 1/2 is 4 - 2A, so the phase is stable below
    # A = 2 and unmixes above it. Nothing here is fitted: the threshold is
    # analytic.
    Amat = Float64[1 1 0; 0 0 1]
    gvec = [0.0, 0.0, 0.0]

    function make_prob(Aex)
        h(x, _) = begin
            N = max(x[2] + x[3], 1.0e-300)
            x2 = max(x[2], 1.0e-300) / N
            x3 = max(x[3], 1.0e-300) / N
            [
                log(max(x[1], 1.0e-300)),
                log(x2) + Aex * x3^2,
                log(x3) + Aex * x2^2,
            ]
        end
        return DualNewtonProblem(
            Amat, gvec, h;
            phases = [
                SolutionPhase([1], 1; always_present = true),
                SolutionPhase([2, 3], 1; mole_fraction = true),
            ],
            idx_bounded = Int[],
        )
    end

    # An equimolar present phase, and the multipliers that make its members
    # stationary there -- which is what equilibrium means for them.
    x_present = [1.0, 0.05, 0.05]

    for (Aex, unmixes) in ((1.0, false), (3.0, true))
        prob = make_prob(Aex)
        u = gvec .+ prob.h(x_present, prob.q0)
        m = phase_split_measure(prob, 2, u, x_present)
        if unmixes
            @test m > 1.0e-6          # a composition below the tangent plane exists
        else
            @test m <= 1.0e-8         # and below the threshold none does
        end
    end

    # A phase of one member cannot split, whatever its model.
    prob1 = DualNewtonProblem(
        Float64[1 0; 0 1], [0.0, 0.0],
        (x, _) -> [log(max(x[1], 1.0e-300)), log(max(x[2], 1.0e-300))];
        phases = [SolutionPhase([2], 1; mole_fraction = true)],
        idx_bounded = Int[],
    )
    @test phase_split_measure(prob1, 1, [0.0, 0.0], [1.0, 1.0]) == -Inf

    # The certificate carries the new fields, and a convex phase AT EQUILIBRIUM
    # is not accused of splitting. Solved first, deliberately: `kkt_certificate`
    # derives its own multipliers, and at a composition that is not a solution
    # they are not the equilibrium ones, so a positive measure there would be
    # correct rather than a false alarm -- and would test nothing.
    prob_ok = make_prob(1.0)
    b_ok = Amat * x_present
    res_ok = dual_newton_solve(prob_ok, b_ok, copy(x_present))
    cert = kkt_certificate(prob_ok, res_ok.x, b_ok)
    @test haskey(cert, :worst_violation_split)
    @test isempty(cert.split_phases)
    @test cert.optimal
end

@testset "a dead end-member is not evidence that a phase wants to split" begin
    # THE REGRESSION 0.5.4 FIXES. `phase_split_measure` starts the successive
    # substitution from each corner of the phase's composition simplex. It did so
    # for every end-member, including one whose conservation row is DEGENERATE --
    # no matter of that component exists in the system, so the row's multiplier is
    # pinned at the sentinel `DEGENERATE_POTENTIAL` rather than solved for.
    #
    # `uᵢ` for such a member is then not a chemical potential and `uᵢ - gᵢ` is a
    # number with no meaning. Read by the measure, it reported a converged,
    # mass-balanced equilibrium as a phase wanting to move to a composition the
    # element balance forbids -- and reported the SAME value on every unrelated
    # system carrying the same declaration, which is what a sentinel does and
    # chemistry does not.
    #
    # Three components, three species: a solvent, and a two-member ideal mixing
    # phase whose SECOND member is the only carrier of component 3. With `b[3]`
    # zero, that member can never form.
    Amat = Float64[1 1 0; 0 0 1; 0 0 1]
    gvec = [0.0, 0.0, 0.0]
    h(x, _) = begin
        N = max(x[2] + x[3], 1.0e-300)
        [
            log(max(x[1], 1.0e-300)),
            log(max(x[2], 1.0e-300) / N),
            log(max(x[3], 1.0e-300) / N),
        ]
    end
    prob = DualNewtonProblem(
        Amat, gvec, h;
        phases = [
            SolutionPhase([1], 1; always_present = true),
            SolutionPhase([2, 3], 1; mole_fraction = true),
        ],
        idx_bounded = Int[],
    )

    x = [1.0, 0.05, 0.0]
    b = Amat * x
    @test b[3] == 0.0                       # component 3 is absent from the system

    dead = Set([3])                          # what `kkt_certificate` derives
    u = gvec .+ h([1.0, 0.05, 0.05], prob.q0)
    # The sentinel is what the solver actually pins such a row's multiplier to,
    # so it is what the measure would read.
    u_sentinel = copy(u)
    u_sentinel[3] = OptimaSolver.DEGENERATE_POTENTIAL

    # Ignoring `dead`, the sentinel drives the measure far positive: this is the
    # defect, reproduced.
    @test phase_split_measure(prob, 2, u_sentinel, x) > 10.0
    # Honoring it, the phase has one live member, so there is no interior to
    # unmix into and no verdict to give.
    @test phase_split_measure(prob, 2, u_sentinel, x; dead = dead) == -Inf
    @test phase_tangent_measure(prob, 2, u_sentinel, x; dead = dead) ==
        phase_tangent_measure(prob, 2, u_sentinel, x; dead = dead, start = [0.5, 0.5])

    # And end to end: the certificate must not fail a converged, mass-balanced
    # equilibrium on account of a member the element balance forbids.
    res = dual_newton_solve(prob, b, copy(x))
    cert = kkt_certificate(prob, res.x, b)
    @test cert.feasibility <= 1.0e-8
    @test isempty(cert.split_phases)
    @test cert.optimal
end

@testset "a solute pinned at a bound is not an unconverged solute" begin
    # `W` is a log-molality clamped to `[-700, 20]`. A species whose stationarity
    # asks for less than the floor can never reach it: the update is clamped, the
    # state does not move, and its residual stays exactly where it is. Counted as
    # a convergence failure, it made the inner inversion run to its sweep cap on
    # every call -- measured on a CEM I at its own converged answer, the worst
    # measure was 1.0054e+02 at sweep 1 and 1.0054e+02 at sweep 200, and it
    # belonged to dissolved oxygen sitting at the floor with 9.9e-305 mol.
    #
    # The condition at an active bound is complementarity, not stationarity.
    Amat = Float64[1 1 1]
    gvec = [0.0, 0.0, 900.0]          # the third species cannot reach its bound
    h(x, _) = [
        log(max(x[1], 1.0e-300)),
        log(max(x[2], 1.0e-300)),
        log(max(x[3], 1.0e-300)),
    ]
    prob = DualNewtonProblem(
        Amat, gvec, h;
        phases = [SolutionPhase([1, 2, 3], 1; always_present = true)],
        idx_bounded = Int[],
    )
    x = [1.0, 0.1, 1.0e-300]
    b = Amat * x
    res = dual_newton_solve(prob, b, copy(x))
    cert = kkt_certificate(prob, res.x, b)

    @test res.x[3] < 1.0e-200         # still at the floor, as it must be
    @test cert.feasibility <= 1.0e-8  # and the solve converged anyway
end

@testset "the inner stopping test measures an error, not a step" begin
    # `_invert_phases!` reports what its sweeps ended on, and the line search
    # uses that to prefer steps whose inner solve converged. The measure it
    # reports is a step size for a mole-fraction phase, and the iteration is
    # linearly convergent, so a small step is not a small error when the
    # contraction rate is close to one.
    #
    # Checked here on the property that matters rather than on an internal:
    # re-running the inversion from its own output must change nothing. That is
    # exactly what `inner_tol` exists to guarantee, and a step-based test does
    # not deliver it.
    Amat = Float64[1 1 0; 0 0 1]
    gvec = [0.0, 0.0, 0.0]
    Aex = 1.0
    h(x, _) = begin
        N = max(x[2] + x[3], 1.0e-300)
        x2 = max(x[2], 1.0e-300) / N
        x3 = max(x[3], 1.0e-300) / N
        [log(max(x[1], 1.0e-300)), log(x2) + Aex * x3^2, log(x3) + Aex * x2^2]
    end
    prob = DualNewtonProblem(
        Amat, gvec, h;
        phases = [
            SolutionPhase([1], 1; always_present = true),
            SolutionPhase([2, 3], 1; mole_fraction = true),
        ],
        idx_bounded = Int[],
    )
    x = [1.0, 0.05, 0.05]
    b = Amat * x
    res = dual_newton_solve(prob, b, copy(x))
    cert = kkt_certificate(prob, res.x, b)
    @test cert.optimal

    # Idempotence: solving again from the answer returns the answer.
    res2 = dual_newton_solve(prob, b, copy(res.x))
    @test maximum(abs, res2.x .- res.x) <= 1.0e-8
end

@testset "an inner iteration that has stopped converging stops sweeping" begin
    # `_invert_phases!` updates a solute as `w ← w + clamp(u − g − h, ±30)`, so an
    # activity model with `h₂ = β ln x₂` gives the map `w ← (1 − β) w` — a real
    # dependence on the composition, not a scripted sequence. Three regimes,
    # and the stall rule must tell them apart.
    function sweeps(β; w0 = 1.0)
        calls = Ref(0)
        h(x, _) = (calls[] += 1; [0.0, β * log(x[2])])
        prob = DualNewtonProblem(
            Float64[1 1], [0.0, 0.0], h;
            phases = [SolutionPhase([1, 2], 1)], idx_bounded = Int[],
        )
        W = [[0.0, w0]]
        resid = Ref(NaN)
        OptimaSolver._invert_phases!(
            prob, W, [0.0], [1.0], [1], Int[], Float64[], zeros(2);
            resid = resid, maxsweeps = 200,
        )
        return (calls = calls[], w = W[1][2], step = resid[])
    end

    # β = 3: `w ← −2w`, an expanding oscillation that the clamp bounds into a
    # cycle between 16 and −14. It never sets a new smallest step after the
    # first sweep, so it stops `INNER_STALL_SWEEPS` sweeps later. Without the
    # rule it ran the full 200, which is what a cement's bad trial points did
    # on 89 % of their calls.
    cyc = sweeps(3.0)
    @test cyc.calls == OptimaSolver.INNER_STALL_SWEEPS + 1
    @test cyc.step == 30.0          # reported as unconverged, as it must be

    # β = 1.5: `w ← −w/2`, a contraction that needs 49 sweeps to reach the
    # threshold — more than `INNER_STALL_SWEEPS`. Its step falls at every sweep,
    # so the rule never fires and it converges exactly as it did before.
    con = sweeps(1.5)
    @test con.calls == 49
    @test abs(con.w) < 1.0e-14

    # β = 0.1: `w ← 0.9w`, converging but too slowly to finish in 200 sweeps.
    # It still sets a new minimum every time, so it runs to the cap as before
    # and is not mistaken for a cycle.
    slow = sweeps(0.1)
    @test slow.calls == 200
    @test 0 < slow.step < 1.0e-9
end

@testset "the trial composition comes back with the verdict" begin
    # `phase_tangent_measure` and `phase_split_measure` answer WHETHER, and a
    # caller that means to act on the answer needs WHICH: the composition the
    # phase wants to move to. It is produced by the same successive substitution
    # that decides the verdict and cannot be recovered afterwards, the trial
    # being a stationary point of the tangent-plane distance in the FULL system.
    #
    # The contract tested here is that the two are the same computation. The
    # `*_measure` functions are defined as `first` of the `*_trial` ones, so this
    # cannot drift, and the test is what says so out loud.
    Amat = Float64[1 1 0; 0 0 1]
    gvec = [0.0, 0.0, 0.0]
    Aex = 3.0
    h(x, _) = begin
        N = max(x[2] + x[3], 1.0e-300)
        x2 = max(x[2], 1.0e-300) / N
        x3 = max(x[3], 1.0e-300) / N
        [
            log(max(x[1], 1.0e-300)),
            log(x2) + Aex * x3^2,
            log(x3) + Aex * x2^2,
        ]
    end
    prob = DualNewtonProblem(
        Amat, gvec, h;
        phases = [
            SolutionPhase([1], 1; always_present = true),
            SolutionPhase([2, 3], 1; mole_fraction = true),
        ],
        idx_bounded = Int[],
    )
    x_present = [1.0, 0.08, 0.02]          # x = 0.2, inside the binodal
    u = gvec .+ h(x_present, prob.q0)

    m, trial = phase_split_trial(prob, 2, u, x_present)
    @test m == phase_split_measure(prob, 2, u, x_present)
    @test m > 1.0e-6                        # it does want to split
    @test length(trial) == 2
    @test all(trial .>= 0)
    @test sum(trial) ≈ 1.0 rtol = 1.0e-8
    # A symmetric regular solution splits symmetrically, so a phase sitting at
    # x = 0.2 is answered by a trial in the OTHER lobe. Nothing here is fitted:
    # the binodal of `x ln x + (1-x) ln(1-x) + A x(1-x)` is the pair (x, 1-x)
    # solving `ln(x/(1-x)) = A(2x-1)`, which for A = 3 is x = 0.0711 / 0.9289.
    @test trial[2] > 0.5

    # Same contract for the absent-phase test, whose trial seeds the same kind
    # of caller.
    mt, tt = phase_tangent_trial(prob, 2, u, x_present)
    @test mt == phase_tangent_measure(prob, 2, u, x_present)
    @test length(tt) == 2
    @test sum(tt) ≈ 1.0 rtol = 1.0e-8
end

@testset "a lobe the corners cannot reach, and the start that reaches it" begin
    # WHY `split_starts` EXISTS, measured rather than asserted.
    #
    # The successive substitution converges to the stationary point of the
    # tangent-plane distance nearest its start. A corner of the composition
    # simplex is a natural start and usually a good one — but it is not always on
    # the right side of the barrier, and where it is not, the iteration walks
    # back to the phase's own composition and reports nothing.
    #
    # The model here is the published AFm sulfate/hydroxide binary of CEMDATA18:
    # Redlich-Kister with a₀ = 0.188 RT and a₁ = 2.49 RT, i.e.
    #
    #     g(x)/RT = x ln x + (1-x) ln(1-x) + x(1-x)[A₀ + A₁(2x-1)] ,
    #
    # whose spinodal is [0.631, 0.914] and whose binodal is [0.4999, 0.9700].
    # Both compositions below are metastable — inside the binodal, outside the
    # spinodal — and the corners find one and miss the other. So being metastable
    # is not what defeats them; sitting in the lobe they lead back into is.
    A0, A1 = 0.188, 2.49
    Gex(n) = begin
        N = n[1] + n[2]
        x = n[2] / N
        N * x * (1 - x) * (A0 + A1 * (2x - 1))
    end
    Amat = Float64[1 1 0; 0 0 1]
    gvec = [0.0, 0.0, 0.0]
    h(x, _) = begin
        n = [max(x[2], 1.0e-300), max(x[3], 1.0e-300)]
        N = n[1] + n[2]
        lg = ForwardDiff.gradient(Gex, n)
        [log(max(x[1], 1.0e-300)), log(n[1] / N) + lg[1], log(n[2] / N) + lg[2]]
    end
    binodal = (0.499905, 0.970026)          # `common_tangent` of that model
    make(starts) = DualNewtonProblem(
        Amat, gvec, h;
        phases = [
            SolutionPhase([1], 1; always_present = true),
            SolutionPhase([2, 3], 1; mole_fraction = true, split_starts = starts),
        ],
        idx_bounded = Int[],
    )
    bare = make(())
    seeded = make([[1 - binodal[1], binodal[1]], [1 - binodal[2], binodal[2]]])

    function verdicts(xp)
        xpres = [1.0, 0.05 * (1 - xp), 0.05 * xp]
        u = gvec .+ bare.h(xpres, bare.q0)
        return (
            phase_split_trial(bare, 2, u, xpres),
            phase_split_trial(seeded, 2, u, xpres),
        )
    end

    # The composition a CEM I actually settles at: the corners find it.
    (m0, t0), (m1, t1) = verdicts(0.5268)
    @test m0 > 1.0e-3
    @test t0[2] > 0.9                       # the trial is in the sulfate-rich lobe
    @test m1 ≈ m0 rtol = 1.0e-6             # a start cannot take a verdict away

    # High on the hydroxide side, still inside the binodal: the corners return
    # to the phase's own composition and report nothing.
    (m0, t0), (m1, t1) = verdicts(0.95)
    @test abs(m0) < 1.0e-10                 # no verdict from the corners
    @test t0[2] ≈ 0.95 rtol = 1.0e-3        # they walked straight back
    @test m1 > 1.0e-3                       # the binodal start finds the split
    @test t1[2] < 0.5                       # and it lies in the other lobe

    # Outside the binodal the phase is genuinely stable, and no start may invent
    # a split: the extra starts are a search, not a bias.
    (m0, _), (m1, _) = verdicts(0.98)
    @test m0 <= 1.0e-10
    @test m1 <= 1.0e-10
end

@testset "simplex_start lands on a vertex of the mass balance" begin
    # GEM-Selektor's initial approximation, after Karpov: minimize the LINEAR
    # part of the Gibbs energy over `A x = b, x ≥ 0`. The optimum of a linear
    # program sits at a vertex, so the answer is a basic feasible solution with
    # at most `m` nonzero species — a poor composition and an exact starting
    # point.
    #
    # 1. A two-species balance with an obvious answer. `x₁ + x₂ = 1` and `x₂`
    #    three times as expensive: everything goes to `x₁`, exactly.
    A = [1.0 1.0]
    x = simplex_start(A, [1.0, 3.0], [1.0])
    @test x !== nothing
    @test A * x ≈ [1.0] rtol = 1.0e-10
    @test x[1] ≈ 1.0 rtol = 1.0e-8
    @test x[2] == 0.0                       # a vertex, not a near-vertex

    # 2. A row whose right-hand side is negative. Phase I needs its artificial
    #    on the correct side, so the row has to be negated first; a solver that
    #    skips that reports infeasible on a system that is not.
    x2 = simplex_start(-A, [1.0, 3.0], [-1.0])
    @test x2 !== nothing
    @test (-A) * x2 ≈ [-1.0] rtol = 1.0e-10

    # 3. Genuinely infeasible: `x₁ + x₂ = -1` with `x ≥ 0`. Phase I cannot drive
    #    the artificial out, and the answer must be `nothing` rather than a
    #    vector that violates the balance it was asked to satisfy.
    @test simplex_start([1.0 1.0], [1.0, 1.0], [-1.0]) === nothing

    # 4. Feasible systems built backwards from a known composition, so
    #    feasibility is not in question and the vertex property is what is
    #    tested. A deterministic generator rather than `Random`: the failure is
    #    then reproducible from the test alone.
    seed = UInt64(0x2026_0914_0000_0001)
    nextrand() = begin
        seed ⊻= seed << 13
        seed ⊻= seed >> 7
        seed ⊻= seed << 17
        return (seed >> 11) / Float64(1 << 53)
    end
    for _ in 1:40
        m = 3 + Int(floor(6 * nextrand()))
        n = 10 + Int(floor(30 * nextrand()))
        Ar = [nextrand() for _ in 1:m, _ in 1:n]
        x0 = zeros(n)
        for _ in 1:m
            x0[1 + Int(floor(n * nextrand()))] = nextrand()
        end
        br = Ar * x0
        gr = [2 * nextrand() - 1 for _ in 1:n]
        xs = simplex_start(Ar, gr, br)
        @test xs !== nothing
        @test norm(Ar * xs - br, Inf) < 1.0e-8       # the balance is EXACT
        @test all(xs .>= 0)
        @test count(>(1.0e-8), xs) <= m              # and the point is a vertex
        # It is also an optimum, so no worse than the composition it was built
        # from — the one property that says the simplex ran rather than returned
        # the first feasible thing it found.
        @test dot(gr, xs) <= dot(gr, x0) + 1.0e-8
    end
end
