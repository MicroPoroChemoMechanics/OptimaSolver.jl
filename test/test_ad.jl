@testset "AD compatibility (ForwardDiff)" begin
    # ── Setup ─────────────────────────────────────────────────────────────────
    # gibbs_hessian_diag is ForwardDiff-compatible
    n0 = [0.5, 0.3, 0.2]
    g_h = ForwardDiff.jacobian(n -> OptimaSolver.gibbs_hessian_diag(n), n0)
    @test all(isfinite, g_h)

    # ── kkt_residual is ForwardDiff-compatible through n ──────────────────────
    μ⁰ = [0.0, 1.0, 2.0]
    A = ones(Float64, 1, 3)
    b = ones(Float64, 1)

    function G(n, p)
        return sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
    end
    function ∇G!(grad, n, p)
        for i in eachindex(n)
            grad[i] = p.μ⁰[i] + log(n[i]) + one(eltype(n))
        end
    end

    prob = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,))

    n_test = [0.6, 0.3, 0.1]
    y_test = [0.5]

    jac_ex = ForwardDiff.jacobian(
        n -> begin
            g = similar(n)
            prob.g!(g, n, prob.p)
            OptimaSolver.kkt_residual(prob, n, y_test, g, 1.0e-4).ex
        end,
        n_test,
    )
    @test all(isfinite, jac_ex)

    # ── hessian_diagonal is ForwardDiff-compatible ────────────────────────────
    h_jac = ForwardDiff.jacobian(
        n -> begin
            hf = OptimaSolver.gibbs_hessian_diag(n)
            OptimaSolver.hessian_diagonal(prob, n, 1.0e-4, hf)
        end,
        n_test,
    )
    @test all(isfinite, h_jac)

    # ── sensitivity matrices are finite ───────────────────────────────────────
    result = solve(prob, OptimaOptions(tol = 1.0e-12))
    @test result.converged

    n_sol = result.n
    hf_sol = OptimaSolver.gibbs_hessian_diag(n_sol)
    h_sol = OptimaSolver.hessian_diagonal(prob, n_sol, 1.0e-14, hf_sol)
    sens = sensitivity(prob, n_sol, result.y, h_sol, 1.0e-14)

    @test all(isfinite, sens.∂n_∂b)
    @test all(isfinite, sens.∂n_∂μ0)

    # ── objective function is ForwardDiff-compatible through μ⁰ ──────────────
    # Differentiate the objective value at the current point w.r.t. μ⁰
    g_obj = ForwardDiff.gradient(
        μ0 -> G(n_test, (μ⁰ = μ0,)),
        μ⁰,
    )
    @test all(isfinite, g_obj)
    @test g_obj ≈ n_test atol = 1.0e-12   # ∂G/∂μ⁰_i = n_i

    # ── a dual-valued `b` drives the whole Newton loop ────────────────────────
    # The tests above differentiate the building blocks only. This one runs the
    # solver itself on duals, which is where `line_search` and `clamp_step`
    # reach for `α_max` and `τ`: annotating either as `Float64` breaks it.
    A2 = [1.0 1.0 0.0; 0.0 1.0 1.0]
    # `1e-10`, not `1e-12`: convergence is judged on the KKT error at μ = 0, and
    # on this two-constraint problem the barrier method reaches 4.6e-12 and no
    # further. Asking for 1e-12 used to succeed only because the error was
    # measured against the current μ, where it vanishes at every barrier level.
    opts_ad = OptimaOptions(tol = 1.0e-10)

    function n₁_of_b₁(x)
        p = OptimaProblem(A2, [x, 1.0], G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,))
        return solve(p, opts_ad).n[1]
    end

    d_ad = ForwardDiff.derivative(n₁_of_b₁, 1.0)
    @test isfinite(d_ad)

    # against the analytic sensitivity, which never sees a dual
    prob_b = OptimaProblem(A2, [1.0, 1.0], G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,))
    res_b = solve(prob_b, opts_ad)
    @test res_b.converged
    hf_b = OptimaSolver.gibbs_hessian_diag(res_b.n)
    h_b = OptimaSolver.hessian_diagonal(prob_b, res_b.n, 1.0e-14, hf_b)
    sens_b = sensitivity(prob_b, res_b.n, res_b.y, h_b, 1.0e-14)
    @test d_ad ≈ sens_b.∂n_∂b[1, 1] rtol = 1.0e-6
end
