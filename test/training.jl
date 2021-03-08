include("imports.jl")


@testset ExtendedTestSet "Training" begin

    @testset ExtendedTestSet "fitonecycle!" begin
        learner = testlearner(Recorder())
        @test_nowarn fitonecycle!(learner, 5)
    end
end
