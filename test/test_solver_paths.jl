@testset "Solver — the exits and fallbacks the toy problem never reaches" begin
    μ⁰ = [0.0, 1.0, 2.0]
    G(n, p) = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
    function ∇G!(grad, n, p)
        for i in eachindex(n)
            grad[i] = p.μ⁰[i] + log(n[i]) + 1
        end
        return nothing
    end
    A = ones(1, 3)
    b = [1.0]
    prob = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,))
    n_analytic = exp.(-μ⁰) ./ sum(exp.(-μ⁰))

    @testset "solve with a pre-built Canonicalizer" begin
        # The documented way to solve many problems that share `A` without
        # redoing the pivoted QR each time. It is public API and nothing
        # exercised it; the point is that it gives the same answer.
        can = OptimaSolver.Canonicalizer(A)
        res_can = solve(prob, can, OptimaOptions(tol = 1.0e-12))
        res_plain = solve(prob, OptimaOptions(tol = 1.0e-12))

        @test res_can.converged
        @test res_can.n ≈ n_analytic atol = 1.0e-7
        @test res_can.n ≈ res_plain.n atol = 1.0e-12
        @test res_can.iterations == res_plain.iterations
    end

    @testset "the barrier runs out before the tolerance is met" begin
        # Asking for a tolerance below what double precision can deliver makes
        # the outer loop exit on `μ ≤ barrier_min` instead of on convergence —
        # a completion path distinct from both success and `max_iter`. What
        # matters is that the reported state is the *final* one and that
        # `converged` says no rather than the solver claiming success.
        res = solve(prob, OptimaOptions(tol = 1.0e-18, max_iter = 500))

        @test !res.converged
        @test res.iterations < 500          # it left on the barrier, not on the count
        @test isfinite(res.error_opt) && isfinite(res.error_feas)
        # It is nevertheless a good answer: failing the tolerance is not the
        # same as failing the problem.
        @test res.n ≈ n_analytic atol = 1.0e-6
        @test abs(only(A * res.n) - only(b)) < 1.0e-10
    end

    @testset "_nnls — Lawson-Hanson non-negative least squares" begin
        # The routine that puts the starting point exactly on `A n = b` with
        # `n ≥ lb`. Reached only when the direct route fails, so no solve in the
        # suite touched it; these drive it directly.
        nnls = OptimaSolver._nnls

        # Attainable: the exact solution is non-negative, so the residual is 0.
        E = [1.0 0.0; 0.0 1.0]
        v = nnls(E, [2.0, 3.0])
        @test v ≈ [2.0, 3.0] atol = 1.0e-12
        @test norm(E * v - [2.0, 3.0]) < 1.0e-12

        # The unconstrained least-squares answer has a negative component, so
        # the active set must clamp it — this is the release loop.
        E2 = [1.0 1.0; 1.0 -1.0]
        f2 = [1.0, 3.0]                     # unconstrained: v = [2, -1]
        v2 = nnls(E2, f2)
        @test all(v2 .>= 0)
        @test v2[2] ≈ 0.0 atol = 1.0e-12
        # Optimal among non-negative vectors: no descent direction remains.
        for w in ([2.0, 0.0], [1.5, 0.0], [2.5, 0.0], [2.0, 0.1])
            @test norm(E2 * v2 - f2) <= norm(E2 * w - f2) + 1.0e-10
        end

        # Overdetermined and infeasible in the exact sense: a genuine residual.
        E3 = reshape([1.0, 1.0, 1.0], 3, 1)
        v3 = nnls(E3, [1.0, 2.0, 3.0])
        @test v3 ≈ [2.0] atol = 1.0e-10

        # A right-hand side pushing every component negative collapses to zero.
        v4 = nnls([1.0 0.0; 0.0 1.0], [-1.0, -2.0])
        @test v4 ≈ [0.0, 0.0] atol = 1.0e-12
    end

    @testset "_project_alternating! — the last-resort feasibility fallback" begin
        # Alternating projections onto `{A n = b}` and onto `{n ≥ lb}`. The
        # docstring is explicit that this is slow and only reached when the
        # exact routes have failed; it still has to work.
        # This point satisfies `A n = b` exactly (0.5 - 0.4 + 0.9 = 1) while
        # sitting below the lower bound. The residual test used to run before
        # the box projection, so the routine returned it untouched — a negative
        # amount handed straight to `log` in the barrier.
        n = [0.5, -0.4, 0.9]
        OptimaSolver._project_alternating!(n, prob; maxit = 500, tol = 1.0e-12)
        @test all(n .>= prob.lb)
        @test abs(only(A * n) - only(b)) < 1.0e-8

        # And the ordinary case: both sets violated at once.
        n_both = [0.2, -0.3, 0.4]
        OptimaSolver._project_alternating!(n_both, prob; maxit = 500, tol = 1.0e-12)
        @test all(n_both .>= prob.lb)
        @test abs(only(A * n_both) - only(b)) < 1.0e-8

        # A point already feasible must come back unchanged.
        n2 = copy(n_analytic)
        OptimaSolver._project_alternating!(n2, prob; maxit = 500, tol = 1.0e-12)
        @test n2 ≈ n_analytic atol = 1.0e-10
    end
end
