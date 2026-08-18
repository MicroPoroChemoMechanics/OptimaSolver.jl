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

    # A two-variable ideal mixture with one linear constraint, whose minimiser is
    # available in closed form. With `f = Σ xᵢ(gᵢ + ln xᵢ)` and `x₁ + x₂ = 1`,
    # stationarity gives `g₁ + ln x₁ = g₂ + ln x₂`, so
    # `x₁ = 1/(1 + e^{g₁-g₂})`.
    g = [0.0, 1.0]
    h(x, _) = log.(max.(x, 1.0e-300))
    A = Float64[1 1]
    b = [1.0]

    prob = DualNewtonProblem(A, g, h; idx_log = [1, 2], idx_bounded = Int[], j_ref = 2)
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

@testset "an empty set of interior variables is refused" begin

    # The multipliers are determined by the stationarity of the strictly positive
    # variables. With none, nothing determines them, and saying so beats
    # returning a number.
    @test_throws ArgumentError DualNewtonProblem(
        Float64[1 1], [0.0, 1.0], (x, _) -> log.(x);
        idx_log = Int[], idx_bounded = [1, 2],
    )

end
