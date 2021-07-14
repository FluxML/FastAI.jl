include("../imports.jl")

@testset ExtendedTestSet "lrfind" begin
    learner = testlearner()
    @test_nowarn result = lrfind(learner)

end
