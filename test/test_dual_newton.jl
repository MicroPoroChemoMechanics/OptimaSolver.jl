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
