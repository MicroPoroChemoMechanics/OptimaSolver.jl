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

    # ── Warm-start: solve again with slightly perturbed μ⁰
    μ⁰2 = [0.0, 0.9, 2.1]
    prob2 = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰2,))

    result2 = solve(prob2, OptimaOptions(tol = 1.0e-12); u0 = result)
    @test result2.converged
    @test result2.iterations < result.iterations + 5  # warm-start helps

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
