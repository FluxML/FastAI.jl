include("../imports.jl")


@testset ExtendedTestSet "fitonecycle!" begin
    learner = testlearner(Recorder())
    @test_nowarn fitonecycle!(learner, 5)
end
