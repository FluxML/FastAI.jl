
struct ConstGrouper <: FastAI.ParamGrouper
    g::Any
end
FastAI.group(cg::ConstGrouper, m) = Dict(cg.g => m)

@testset "finetune!" begin
    learner = testlearner(Recorder())
    @test_nowarn finetune!(learner, 1; grouper = ConstGrouper(2))
end

@testset "fitonecycle!" begin
    learner = testlearner(Recorder())
    @test_nowarn fitonecycle!(learner, 5)
end

@testset "lrfind" begin
    learner = testlearner()
    @test_nowarn result = lrfind(learner)
end
