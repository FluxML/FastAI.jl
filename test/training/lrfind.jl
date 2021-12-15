@testset "lrfind" begin
    learner = testlearner()
    @test_nowarn result = lrfind(learner)
end
