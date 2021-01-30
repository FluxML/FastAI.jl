include("imports.jl")


@testset ExtendedTestSet "Training" begin

    @testset ExtendedTestSet "fitonecycle!" begin
        learner = testlearner(Recorder())
        fitonecycle!(learner, 5)
    end
end
