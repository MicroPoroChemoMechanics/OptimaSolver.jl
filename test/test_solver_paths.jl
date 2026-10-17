@testset "Solver — the exits and fallbacks the toy problem never reaches" begin
    μ⁰ = [0.0, 1.0, 2.0]
    G(n, p) = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
    ∇G!(grad, n, p) = grad .= p.μ⁰ .+ log.(n) .+ 1
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
        # the active set must clamp it. Here the clamped column never enters:
        # its gradient component is negative from the start.
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

        # The release step. Column 2 enters first (v = [0, 0.8]), then column 1,
        # and their joint least-squares solution [3, -1] is negative in column 2:
        # the iterate steps back to where column 2 reaches zero and lets it go.
        # What remains is the projection on column 1, (1 + 2)/2 = 1.5.
        E5 = [1.0 2.0; 1.0 1.0]
        f5 = [1.0, 2.0]
        v5 = nnls(E5, f5)
        @test v5 ≈ [1.5, 0.0] atol = 1.0e-12
        # Lawson-Hanson's optimality conditions: the gradient of ½‖Ev − f‖² vanishes
        # on the support and points outward off it.
        w5 = E5' * (f5 - E5 * v5)
        @test abs(w5[1]) < 1.0e-12 && w5[2] < 0
    end

    @testset "_initialise_feasible! — the exact routes after the least disturbance" begin
        # The least-disturbance route alternates a minimum-norm correction with a
        # lift above the bounds, fifty times. On these two budgets it is still
        # off the affine set after them -- by 2.1e-3 and by 3.2e-6, against
        # thresholds of 7.5e-8 and 4.5e-8 -- so the exact routes take over.
        f(n, _) = sum(abs2, n)
        g!(gr, n, _) = (gr .= 2 .* n)
        for (A, b, n0, margin) in (
                # the basic/non-basic split: one solve for the basic amounts
                ([2.0 3.0 3.0 3.0; 2.0 0.0 0.0 1.0], [5.0, 3.0], [-3.0, 0.0, -3.0, 4.0], 0.0),
                # NNLS on the slacks, kept above the bounds by its margin δ
                ([3.0 0.0 3.0 1.0; 2.0 2.0 3.0 0.0], [3.0, 2.0], [-2.0, 2.0, 1.0, -1.0], 1.0e-6),
            )
            p = OptimaProblem(A, b, f, g!)
            n = copy(n0)
            OptimaSolver._initialise_feasible!(n, p, OptimaSolver.Canonicalizer(A))
            @test maximum(abs, A * n .- b) < 1.0e-12
            @test all(n .> p.lb)
            @test minimum(n .- p.lb) >= margin
        end
    end

    @testset "the exact Hessian diagonal handed over through the parameters" begin
        # `p.hdiag` replaces both approximations of ∂²f/∂nᵢ²; for this ideal
        # mixture it is 1/nᵢ exactly. The answer is the same closed form.
        calls = Ref(0)
        hdiag(hf, n) = (calls[] += 1; hf .= 1 ./ n)
        prob_h = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰, hdiag = hdiag))
        res = solve(prob_h, OptimaOptions(tol = 1.0e-12))
        @test res.converged
        @test res.n ≈ n_analytic atol = 1.0e-9
        @test calls[] > 0
    end

    @testset "OptimaState from its sizes or from a point" begin
        s = OptimaState(3, 2)
        @test s.n == ones(3) && s.y == zeros(2) && s.μ == 1.0e-4
        @test s.iter == 0 && !s.converged && isinf(s.error_opt)
        @test OptimaState(2, 1, Float32) isa OptimaState{Float32}
        n0, y0 = [0.2, 0.3, 0.5], [1.0]
        s2 = OptimaState(n0, y0, 1.0e-6)
        @test s2.n == n0 && s2.n !== n0          # a copy, not the caller's vector
        @test s2.y == y0 && s2.μ == 1.0e-6 && !s2.converged
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
