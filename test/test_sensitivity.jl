@testset "Sensitivity ∂n*/∂b and ∂n*/∂μ⁰" begin
    # Same 1-row ideal Gibbs problem as test_solver
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
    result = solve(prob, OptimaOptions(tol = 1.0e-12))
    @test result.converged

    # Build Hessian diagonal at the solution
    n = result.n
    hf = OptimaSolver.gibbs_hessian_diag(n)
    # barrier at convergence: μ is very small, use eps as proxy
    μ_conv = result.iterations > 0 ? 1.0e-14 : 1.0e-14
    h = OptimaSolver.hessian_diagonal(prob, n, μ_conv, hf)

    sens = sensitivity(prob, n, result.y, h, μ_conv)

    @test size(sens.∂n_∂b) == (3, 1)
    @test size(sens.∂n_∂μ0) == (3, 3)
    @test all(isfinite, sens.∂n_∂b)
    @test all(isfinite, sens.∂n_∂μ0)

    # ∂n*/∂b: increasing total moles should increase all nᵢ proportionally
    @test all(sens.∂n_∂b .> 0)
    @test sum(sens.∂n_∂b[:, 1]) ≈ 1.0 atol = 1.0e-8  # Σ ∂nᵢ/∂b = 1

    # ── Checked against the CLOSED FORM, not against a difference of solves ──
    #
    # This problem has an exact solution: stationarity gives
    # `nᵢ = b e^{-μ⁰ᵢ} / Σⱼ e^{-μ⁰ⱼ}`, hence
    #
    #     ∂nᵢ/∂b   = nᵢ / b
    #     ∂nᵢ/∂μ⁰ₖ = nᵢ (nₖ/b − δᵢₖ)
    #
    # A finite difference across two solves is the wrong instrument here and was
    # the wrong instrument before: it divides the solver's error by the step, so
    # with `δb = 1e-5` and solves accurate to 1e-10 the quotient carries 1e-5 of
    # noise — exactly the tolerance it was asked to meet. It passed by
    # coincidence, through error cancellation between two neighboring warm-started
    # solves, and any change to the iteration path flipped it while the analytic
    # sensitivity stayed right to nine digits.
    S = sum(exp.(-μ⁰))
    n_exact = b[1] .* exp.(-μ⁰) ./ S
    @test result.n ≈ n_exact rtol = 1.0e-8

    ∂n_∂b_exact = n_exact ./ b[1]
    @test sens.∂n_∂b[:, 1] ≈ ∂n_∂b_exact rtol = 1.0e-6

    ∂n_∂μ0_exact = [
        n_exact[i] * (n_exact[k] / b[1] - (i == k)) for i in 1:3, k in 1:3
    ]
    @test sens.∂n_∂μ0 ≈ ∂n_∂μ0_exact rtol = 1.0e-6

    # A finite difference is still worth having as a coarse cross-check, with a
    # step large enough that the solver error is not what it measures.
    δb = 1.0e-2
    prob_pert = OptimaProblem(A, b .+ δb, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,))
    result_pert = solve(prob_pert, OptimaOptions(tol = 1.0e-12); u0 = result)
    ∂n_∂b_fd = (result_pert.n .- result.n) ./ δb
    @test sens.∂n_∂b[:, 1] ≈ ∂n_∂b_fd atol = 1.0e-6
end
