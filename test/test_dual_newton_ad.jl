# Differentiating through `dual_newton_solve`: a budget or data carrying dual
# numbers, the derivatives from the implicit-function theorem at the answer, and
# nested differentiations kept apart by their tags. Every reference is analytic.

@testset "derivatives through the dual Newton" begin
    # One ideal mole-fraction phase: x = β f, f = softmax(−g).
    A = ones(1, 3)
    g = [0.0, 1.0, 2.0]
    hmf(x, _) = (N = sum(x); [log(max(x[i], 1.0e-300) / N) for i in 1:3])
    phases = [SolutionPhase([1, 2, 3], 1; always_present = true, mole_fraction = true)]
    prob = DualNewtonProblem(A, g, hmf; phases = phases)
    x0 = [0.4, 0.3, 0.3]
    f = exp.(-g) ./ sum(exp.(-g))
    β = 2.0

    @testset "the budget" begin
        dx = ForwardDiff.derivative(bb -> dual_newton_solve(prob, [bb], x0).x, β)
        @test dx ≈ f rtol = 1.0e-10
        # x is linear in β: its second derivative vanishes, and the two
        # differentiations nest under distinct tags.
        d2 = ForwardDiff.derivative(bb -> ForwardDiff.derivative(c -> dual_newton_solve(prob, [c], x0).x[1], bb), β)
        @test abs(d2) < 1.0e-9
    end

    @testset "the standard potentials" begin
        J = ForwardDiff.jacobian(gg -> dual_newton_solve(DualNewtonProblem(A, gg, hmf; phases = phases), [β], x0).x, g)
        @test J ≈ β .* (f * transpose(f) .- Diagonal(f)) rtol = 1.0e-9
        d2 = ForwardDiff.derivative(
            a -> ForwardDiff.derivative(
                c -> dual_newton_solve(DualNewtonProblem(A, [c, 1.0, 2.0], hmf; phases = phases), [β], x0).x[1], a,
            ),
            0.0,
        )
        @test d2 ≈ β * f[1] * (1 - f[1]) * (1 - 2f[1]) rtol = 1.0e-8
    end

    @testset "the parameters of the activity model" begin
        # A shift `c` of the first member's log activity carried by `params`:
        # it acts as a shift of `g₁`, so `dx/dc` is the first column above.
        hc(x, p) = (N = sum(x); [log(max(x[i], 1.0e-300) / N) + (i == 1 ? p.c : zero(p.c)) for i in 1:3])
        dx = ForwardDiff.derivative(
            c -> dual_newton_solve(DualNewtonProblem(A, g, hc; phases = phases, params = (c = c,)), [β], x0).x, 0.0,
        )
        @test dx ≈ β .* (f .* f[1] .- [f[1], 0.0, 0.0]) rtol = 1.0e-9
    end

    @testset "a solvent phase" begin
        # Water and two solutes of one element: x₁/x₂ = e^{g₂ − g₁}, whatever
        # the water. The solutes go through the Newton of the block and the
        # outer Jacobian through the implicit-function theorem.
        Aw = Float64[1 0 0; 0 1 1]
        gw = [0.0, 0.5, 1.5]
        hw(x, _) = [log(x[1] / sum(x)), log(max(x[2], 1.0e-300) / x[1]), log(max(x[3], 1.0e-300) / x[1])]
        pw = DualNewtonProblem(Aw, gw, hw; phases = [SolutionPhase([1, 2, 3], 1; always_present = true)])
        J = ForwardDiff.jacobian(bb -> dual_newton_solve(pw, bb, [55.0, 0.01, 0.01]).x, [55.5, 0.1])
        fs = exp.(-gw[2:3]) ./ sum(exp.(-gw[2:3]))
        @test J[2:3, 2] ≈ fs rtol = 1.0e-8
        @test maximum(abs, J[2:3, 1]) < 1.0e-10
        @test J[1, :] ≈ [1.0, 0.0] atol = 1.0e-10
    end

    @testset "the tangent at an answer obtained elsewhere" begin
        # Water, a solute and its pure solid. Present, the solid fixes the
        # solute (x₂ = x₁ e^{g₃ − g₂}) and takes every mole added; absent, the
        # solute takes them. The active set and the multipliers are recovered
        # from the answer alone.
        A3 = Float64[1 0 0; 0 1 1]
        h3(x, _) = [log(x[1] / (x[1] + x[2])), log(max(x[2], 1.0e-300) / x[1]), 0.0]
        b3 = [55.5, 0.1]
        for (g3, present) in ((-10.0, true), (-2.0, false))
            p3 = DualNewtonProblem(
                A3, [0.0, 0.0, g3], h3;
                phases = [SolutionPhase([1, 2], 1; always_present = true)], idx_bounded = [3],
            )
            res = dual_newton_solve(p3, b3, [55.5, 0.05, 0.05])
            @test (res.x[3] > 0) == present
            J = ForwardDiff.jacobian(bb -> dual_newton_tangent(p3, bb, res.x).x, b3)
            r = res.x[2] / res.x[1]
            if present
                @test J[:, 2] ≈ [0.0, 0.0, 1.0] atol = 1.0e-10
                @test J[:, 1] ≈ [1.0, r, -r] rtol = 1.0e-8
            else
                @test J[:, 2] ≈ [0.0, 1.0, 0.0] atol = 1.0e-10
                @test J[3, 1] == 0
            end
            # The same as the derivative through the solve, which knows its
            # active set and its multipliers.
            @test J ≈ ForwardDiff.jacobian(bb -> dual_newton_solve(p3, bb, [55.5, 0.05, 0.05]).x, b3) rtol = 1.0e-10
        end
        # A problem whose data carry no dual number has nothing to lift.
        p0 = DualNewtonProblem(A, g, hmf; phases = phases)
        @test_throws ArgumentError dual_newton_tangent(p0, [β], dual_newton_solve(p0, [β], x0).x)
    end

    @testset "a certificate is a verdict on values" begin
        # The answer, the budget and the problem on dual numbers, nested at that:
        # the certificate is that of their values, whatever they carry.
        p0 = DualNewtonProblem(A, g, hmf; phases = phases)
        c0 = kkt_certificate(p0, dual_newton_solve(p0, [β], x0).x, [β])
        verdict(gg, bb) = (
            pd = DualNewtonProblem(A, gg, hmf; phases = phases);
            kkt_certificate(pd, dual_newton_solve(pd, [bb], x0).x, [bb])
        )
        T1 = typeof(ForwardDiff.Tag(:outer, Float64))
        T2 = typeof(ForwardDiff.Tag(:inner, ForwardDiff.Dual{T1, Float64, 1}))
        c = ForwardDiff.Dual{T2}(ForwardDiff.Dual{T1}(g[1], 1.0), ForwardDiff.Dual{T1}(1.0, 0.0))
        cd = verdict([c, g[2], g[3]], ForwardDiff.Dual{T1}(β, 1.0))
        @test cd.optimal == c0.optimal == true
        @test cd.feasibility isa Float64 && cd.feasibility == c0.feasibility
        # And a dual number prints as its value in the iteration log.
        @test OptimaSolver._fmt_sci(ForwardDiff.Dual(1.23456, 7.0)) == "1.23"
    end
end

@testset "a verdict on values, whatever the callbacks carry" begin
    # The two-phase problem of "the certificate tests a mixing phase held
    # absent", its `h` capturing a dual shift of the carrier's activity. The
    # certificate is taken on values: until 0.8.2 the tangent-plane measure of the
    # absent phase wrote the dual into a plain buffer and raised.
    A = Float64[1 1 0; 0 0 1]
    g = [0.0, 0.0, 0.0]
    hshift(c) = (x, _) -> begin
        N = max(x[2] + x[3], 1.0e-300)
        [log(max(x[1], 1.0e-300)) + c, log(max(x[2], 1.0e-300) / N), log(max(x[3], 1.0e-300) / N)]
    end
    phases = [SolutionPhase([1], 1; always_present = true), SolutionPhase([2, 3], 1; mole_fraction = true)]
    plain = DualNewtonProblem(A, g, hshift(0.0); phases = phases)
    dual = DualNewtonProblem(A, g, hshift(ForwardDiff.Dual(0.0, 1.0)); phases = phases)
    x_absent = [1.0, 0.0, 0.0]
    u = -(transpose(A) * [0.0, -5.0])
    @test phase_tangent_measure(dual, 2, u, x_absent) == phase_tangent_measure(plain, 2, u, x_absent)
    @test phase_tangent_measure(plain, 2, ForwardDiff.Dual.(u, 1.0), x_absent) ==
        phase_tangent_measure(plain, 2, u, x_absent)
    cd, cp = kkt_certificate(dual, x_absent, [1.0, 0.0]), kkt_certificate(plain, x_absent, [1.0, 0.0])
    @test cd.absent_phases == cp.absent_phases
    @test cd.worst_violation == cp.worst_violation
    @test cd.optimal == cp.optimal

    # Starts of the split search computed from a model being differentiated: a
    # place to start from, taken by value. Until 0.8.2 the phase refused them.
    sp = SolutionPhase([2, 3], 1; mole_fraction = true, split_starts = [ForwardDiff.Dual.([0.3, 0.7], 1.0)])
    @test sp.split_starts == [[0.3, 0.7]]
    pstart = DualNewtonProblem(A, g, hshift(0.0); phases = [phases[1], sp])
    @test phase_split_trial(pstart, 2, u, [0.5, 0.2, 0.3]; starts = (ForwardDiff.Dual.([0.6, 0.4], 2.0),)) ==
        phase_split_trial(pstart, 2, u, [0.5, 0.2, 0.3]; starts = ([0.6, 0.4],))
end

@testset "duals captured by gq or cq are named, as those of h are" begin
    # They cannot be stripped by the solve. Until 0.8.2 they failed on the first
    # plain buffer with a `MethodError` that named nothing; now they are refused
    # by name, with the two routes that work.
    A = Float64[1 1]
    h(x, _) = log.(max.(x, 1.0e-300))
    mk(gq, cq) = DualNewtonProblem(
        A, [0.0, 0.0], h; phases = [SolutionPhase([1, 2], 2; always_present = true)], gq, cq, q0 = [0.1],
    )
    d = ForwardDiff.Dual(0.6, 1.0)
    @test_throws "`cq` returns dual numbers" dual_newton_solve(
        mk((q, _) -> [0.0, q[1]], (x, q, _) -> [x[1] - d]), [1.0], [0.5, 0.5],
    )
    @test_throws "`gq` returns dual numbers" dual_newton_solve(
        mk((q, _) -> [0.0, q[1] + d], (x, q, _) -> [x[1] - 0.6]), [1.0], [0.5, 0.5],
    )
end
