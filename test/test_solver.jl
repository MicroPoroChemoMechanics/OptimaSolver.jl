@testset "Solver — toy Gibbs problem" begin
    # ── Problem: minimize G(n) = Σ nᵢ (μᵢ⁰ + ln nᵢ)
    #    subject to  A n = b,  n ≥ ε
    #
    # For an ideal solution this has the analytic solution
    #   nᵢ* = exp(λ - μᵢ⁰) / Z   where λ = Lagrange multiplier enforcing Σ nᵢ = b
    #
    # We use a 1-row system: [1 1 1] n = b (total moles = b[1])
    # with μ⁰ = [0, 1, 2] so the analytic solution is
    #   nᵢ* ∝ exp(-μᵢ⁰) = [1, e⁻¹, e⁻²]
    # normalized to sum = b[1] = 1.

    μ⁰ = [0.0, 1.0, 2.0]

    function G(n, p)
        μ0 = p.μ⁰
        return sum(n[i] * (μ0[i] + log(n[i])) for i in eachindex(n))
    end

    function ∇G!(grad, n, p)
        μ0 = p.μ⁰
        for i in eachindex(n)
            grad[i] = μ0[i] + log(n[i]) + 1
        end
    end

    A = ones(1, 3)
    b = [1.0]
    prob = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,))

    result = solve(prob, OptimaOptions(tol = 1.0e-12, verbose = false))

    @test result.converged
    @test norm(A * result.n .- b) < 1.0e-10
    @test all(result.n .> 0)

    # Analytic solution
    n_analytic = exp.(-μ⁰)
    n_analytic ./= sum(n_analytic)
    @test result.n ≈ n_analytic atol = 1.0e-7

    # ── Warm-start: solve a perturbed problem from the previous answer
    μ⁰2 = [0.0, 0.9, 2.1]
    prob2 = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰2,))

    # The control is THE SAME problem solved cold, and not — as this asserted — the
    # iteration count of the PREVIOUS problem. Two different problems need not cost
    # the same, so that comparison could come out either way for reasons having
    # nothing to do with warm starting; on Julia 1.13.0-rc3 it came out `52 < 51`
    # and failed while the warm start was in fact working.
    #
    # The tolerance is also 1e-10 rather than 1e-12. Asking twelve digits of an
    # objective of size 0.6 drives both solves into the regime where the Armijo test
    # is settled by rounding rather than by the function — see the note in
    # `line_search` — and the iteration count there measures that noise instead of
    # the distance from the starting point. Measured on this problem: at 1e-12 the
    # warm start takes 52 iterations against 39 from cold, while at 1e-10, 1e-8 and
    # 1e-6 it saves a consistent four to five (here 21 against 26).
    opts_ws = OptimaOptions(tol = 1.0e-10)
    result2 = solve(prob2, opts_ws; u0 = result)
    result2_cold = solve(prob2, opts_ws)

    n_analytic2 = exp.(-μ⁰2)
    n_analytic2 ./= sum(n_analytic2)

    @test result2.converged
    @test result2.n ≈ n_analytic2 atol = 1.0e-7
    @test result2.iterations < result2_cold.iterations   # the warm start helps

    # ── 2-constraint problem: 2 elements, 4 species
    #    A = [2 1 1 2; 1 0 1 0],  b = [4, 1]
    A2 = Float64[2 1 1 2; 1 0 1 0]
    b2 = [4.0, 1.0]
    μ⁰4 = [0.0, 0.5, 1.0, 1.5]

    function G4(n, p)
        μ0 = p.μ⁰
        return sum(n[i] * (μ0[i] + log(n[i])) for i in eachindex(n))
    end

    function ∇G4!(grad, n, p)
        μ0 = p.μ⁰
        for i in eachindex(n)
            grad[i] = μ0[i] + log(n[i]) + 1
        end
    end

    prob4 = OptimaProblem(A2, b2, G4, ∇G4!; lb = fill(1.0e-16, 4), p = (μ⁰ = μ⁰4,))
    result4 = solve(prob4, OptimaOptions(tol = 1.0e-10))

    @test result4.converged
    @test norm(A2 * result4.n .- b2) < 1.0e-8
    @test all(result4.n .> 0)
end
