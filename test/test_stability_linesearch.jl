@testset "Stability classification and line-search filter" begin
    # Small pure routines the toy solves never drive into their interesting
    # branch: nothing in the suite had a variable pinned at its bound with a
    # step pushing it further down, nor a filter that rejects a candidate.

    @testset "stability_measure" begin
        n = [1.0, 2.0, 3.0]
        lb = [0.0, 1.0, 3.0]
        ex = [0.5, -4.0, 7.0]
        # sᵢ = (nᵢ - lbᵢ)·|exᵢ|; the sign of the residual does not enter.
        @test OptimaSolver.stability_measure(n, lb, ex) ≈ [0.5, 4.0, 0.0]
        # A variable sitting exactly on its bound measures zero however large
        # its residual — that is the point of the measure.
        @test OptimaSolver.stability_measure([1.0], [1.0], [1.0e6])[1] == 0.0
    end

    @testset "classify_variables" begin
        # Slack: [1.0, 1e-12, 1e-12]; threshold = 1e-8 × 1.0.
        n = [2.0, 1.0e-12, 1.0e-12]
        lb = [1.0, 0.0, 0.0]
        # Variable 2 is at its bound with the residual pushing it down
        # (ex ≥ 0) → unstable. Variable 3 is at its bound too, but the residual
        # points away (ex < 0) → stable.
        js, ju = OptimaSolver.classify_variables(n, lb, [0.0, 1.0, -1.0])
        @test js == [1, 3]
        @test ju == [2]
        # The two sets always partition the indices.
        @test sort(vcat(js, ju)) == collect(eachindex(n))
    end

    @testset "reduced_step_for_unstable!" begin
        # A downward step on an unstable variable is capped at τ × slack, so the
        # variable can consume at most that fraction of its remaining room.
        n = [1.0, 2.0]
        lb = [0.0, 0.0]
        dn = [-10.0, -10.0]
        OptimaSolver.reduced_step_for_unstable!(dn, [1], n, lb; τ = 0.5)
        @test dn[1] == -0.5           # capped: -τ × slack = -0.5 × 1.0
        @test dn[2] == -10.0          # index 2 is not in `ju`, untouched

        # A step already smaller than the cap is left alone, and an upward step
        # is never touched — nothing pushes such a variable toward its bound.
        dn2 = [-0.1, 3.0]
        OptimaSolver.reduced_step_for_unstable!(dn2, [1, 2], n, lb; τ = 0.5)
        @test dn2 == [-0.1, 3.0]
    end

    @testset "line-search filter" begin
        f = OptimaSolver.LineSearchFilter(Float64)
        @test isempty(f.entries)
        # An empty filter accepts anything.
        @test OptimaSolver.is_acceptable(f, 1.0, 1.0)

        OptimaSolver.add_to_filter!(f, 1.0, 1.0)
        @test length(f.entries) == 1
        # Dominated in both measures → rejected. Wächter & Biegler's rule: a
        # candidate no better in feasibility AND no better in objective is not
        # worth a step.
        @test !OptimaSolver.is_acceptable(f, 2.0, 2.0)
        @test !OptimaSolver.is_acceptable(f, 1.0, 1.0)
        # Better in either measure → accepted.
        @test OptimaSolver.is_acceptable(f, 0.5, 2.0)
        @test OptimaSolver.is_acceptable(f, 2.0, 0.5)
    end
end
