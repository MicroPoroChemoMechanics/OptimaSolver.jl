# The composition of a mixing phase by Newton's method (`SolutionPhase(...;
# newton = true)`), and members that may be absent from a present phase
# (`bounded_members`). The toy is ideal mixing on two sites of multiplicity `m`,
# as in the ternary tobermorite model of C-S-H: P puts `a` on both sites, Q puts
# `b` on the first and `a` on the second, R puts `b` on both. Q owns no species
# of its own, so its activity stays finite as it vanishes.

using LinearAlgebra

@testset "Newton inversion of a sublattice phase" begin
    safelog(v) = log(max(v, 1.0e-300))
    function sites(x, m)
        N = x[1] + x[2] + x[3]
        f = x ./ N
        s1a, s1b = f[1], f[2] + f[3]
        s2a, s2b = f[1] + f[2], f[3]
        return [
            m * (safelog(s1a) + safelog(s2a)),
            m * (safelog(s1b) + safelog(s2a)),
            m * (safelog(s1b) + safelog(s2b)),
        ]
    end
    A = Float64[2 1 0; 0 1 2]
    b = [1.0, 1.0]
    m = 3.0
    h(x, _) = sites(x, m)
    phase(; kw...) = [SolutionPhase([1, 2, 3], 1; always_present = true, mole_fraction = true, kw...)]

    # The minimum, independently: the balance leaves one direction free,
    # n(t) = (0.5 + t, −2t, 0.5 + t) for t in [−0.5, 0], and G is convex along it.
    function G(t, g)
        n = [0.5 + t, -2t, 0.5 + t]
        tot = sum(n)
        f = n ./ tot
        s = [f[1], f[2] + f[3], f[1] + f[2], f[3]]
        occ = [n[1], n[2] + n[3], n[1] + n[2], n[3]]
        return dot(g, n) + m * sum(o > 0 ? o * log(fs) : 0.0 for (o, fs) in zip(occ, s))
    end
    function argmin_t(g)
        lo, hi = -0.5 + 1.0e-15, -1.0e-15
        for _ in 1:200
            a, c = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            G(a, g) < G(c, g) ? (hi = c) : (lo = a)
        end
        return (lo + hi) / 2
    end

    @testset "the substitution fails where Newton converges" begin
        g = [0.0, -1.0, 0.0]
        x0 = [0.4, 0.2, 0.4]
        sub = dual_newton_solve(DualNewtonProblem(A, g, h; phases = phase()), b, x0)
        newton = dual_newton_solve(DualNewtonProblem(A, g, h; phases = phase(newton = true)), b, x0)
        @test !sub.converged
        @test newton.converged
        prob = DualNewtonProblem(A, g, h; phases = phase(newton = true))
        @test kkt_certificate(prob, newton.x, b).optimal
        t = argmin_t(g)
        @test newton.x ≈ [0.5 + t, -2t, 0.5 + t] rtol = 1.0e-7
    end

    @testset "a phase-local h gives the same answer" begin
        g = [0.0, -1.0, 0.0]
        x0 = [0.4, 0.2, 0.4]
        whole = dual_newton_solve(DualNewtonProblem(A, g, h; phases = phase(newton = true)), b, x0)
        own = dual_newton_solve(
            DualNewtonProblem(A, g, h; phases = phase(newton = true, local_h = xm -> sites(xm, m))),
            b, x0,
        )
        @test own.converged
        @test own.x ≈ whole.x rtol = 1.0e-12
    end

    @testset "on an ideal phase the two agree" begin
        hi(x, _) = [log(max(x[i], 1.0e-300) / max(sum(x), 1.0e-300)) for i in 1:3]
        g = [0.0, -0.3, 0.2]
        x0 = [0.4, 0.2, 0.4]
        r1 = dual_newton_solve(DualNewtonProblem(A, g, hi; phases = phase()), b, x0)
        r2 = dual_newton_solve(DualNewtonProblem(A, g, hi; phases = phase(newton = true)), b, x0)
        @test r1.converged && r2.converged
        @test r2.x ≈ r1.x rtol = 1.0e-10
        # And the tangent-plane measure is the same number both ways.
        u = -(transpose(A) * [0.3, -0.2])
        for newton in (false, true)
            p = DualNewtonProblem(A, g, hi; phases = [SolutionPhase([1, 2, 3], 1; mole_fraction = true, newton)])
            @test phase_tangent_measure(p, 1, u, zeros(3)) ≈ log(sum(exp.(u .- g))) atol = 1.0e-10
        end
    end

    @testset "a bounded member absent from a present phase is tested, not excluded" begin
        # Q unstable: it leaves the phase, which stays present.
        g = [0.0, 50.0, 0.0]
        prob = DualNewtonProblem(A, g, h; phases = phase(newton = true, bounded_members = [2]))
        r = dual_newton_solve(prob, b, [0.4, 0.2, 0.4])
        @test r.converged
        @test r.x[2] < 1.0e-25
        @test r.x ≈ [0.5, 0.0, 0.5] atol = 1.0e-12
        @test kkt_certificate(prob, r.x, b).optimal
        # Q very stable, and a composition that omits it anyway. Declared
        # bounded, it is tested by the inequality of a bounded member and
        # refused. Not declared, it is a member below the floor: it passed until
        # 0.7.2, excluded from every test, and is now held to the one-sided
        # stationarity of such a member, and refused too.
        g = [0.0, -20.0, 0.0]
        x = [0.5, 0.0, 0.5]
        loose = DualNewtonProblem(A, g, h; phases = phase(newton = true))
        strict = DualNewtonProblem(A, g, h; phases = phase(newton = true, bounded_members = [2]))
        l = kkt_certificate(loose, x, b)
        @test !l.optimal && l.stationarity_floored > 1
        c = kkt_certificate(strict, x, b)
        @test !c.optimal
        @test c.worst_violation > 1
    end

    @testset "a bounded member barely unstable leaves the phase exactly" begin
        # With P and R at one half each, the multipliers make Q's condition
        # g_Q ≥ 0: for a small positive g_Q it is absent by that margin. Left to
        # the Newton step it stopped at 1e-16 to 1e-22, above the floor of the
        # certificate, which then judged it as a present member and refused it,
        # by a stationarity proportional to g_Q.
        for gq in (5.0, 0.5, 0.05, 0.005), x0 in ([0.4, 0.2, 0.4], [0.45, 0.1, 0.45])
            g = [0.0, gq, 0.0]
            prob = DualNewtonProblem(A, g, h; phases = phase(newton = true, bounded_members = [2]))
            r = dual_newton_solve(prob, b, x0)
            @test r.converged
            @test r.x[2] < 1.0e-300
            @test r.x ≈ [0.5, 0.0, 0.5] atol = 1.0e-12
            @test kkt_certificate(prob, r.x, b).optimal
        end
        # Barely stable, it stays, at the minimum of the energy along the one
        # direction the balance leaves free.
        g = [0.0, -0.05, 0.0]
        prob = DualNewtonProblem(A, g, h; phases = phase(newton = true, bounded_members = [2]))
        r = dual_newton_solve(prob, b, [0.4, 0.2, 0.4])
        @test r.converged && kkt_certificate(prob, r.x, b).optimal
        t = argmin_t(g)
        @test r.x ≈ [0.5 + t, -2t, 0.5 + t] rtol = 1.0e-7
    end

    @testset "what the keywords refuse" begin
        @test_throws ArgumentError SolutionPhase([1, 2], 1; newton = true)
        @test_throws ArgumentError SolutionPhase([1, 2], 1; mole_fraction = true, bounded_members = [3])
    end
end

@testset "a present phase pinned by its own row at a trace total" begin
    # A surface site family: a free site S and a complex C share a budget N
    # (row 1), C taking one X from a pure species (row 2). Ideal mixing on the
    # sites, so C/S = K whatever N: S = N/(1 + K), C = N K/(1 + K). The phase is
    # always present, its total pinned by row 1, and its total was seeded at no
    # less than 1e-6: at N = 1e-9 the search did not come back down to it.
    A = Float64[1 1 0; 0 1 1]
    K = 4.0
    g = [0.0, -log(K), 0.0]
    hs(x, _) = [
        log(max(x[1], 1.0e-300) / max(x[1] + x[2], 1.0e-300)),
        log(max(x[2], 1.0e-300) / max(x[1] + x[2], 1.0e-300)),
        0.0,
    ]
    sitephase = [SolutionPhase([1, 2], 1; always_present = true, mole_fraction = true)]
    for N in (1.0, 1.0e-6, 1.0e-9, 1.0e-10)
        prob = DualNewtonProblem(A, g, hs; phases = sitephase, idx_bounded = [3])
        b = [N, 1.0]
        r = dual_newton_solve(prob, b, [N, 0.0, 1.0])
        @test r.converged
        @test r.x[1] ≈ N / (1 + K) rtol = 1.0e-8
        @test r.x[2] ≈ N * K / (1 + K) rtol = 1.0e-8
        @test kkt_certificate(prob, r.x, b).optimal
    end
end
