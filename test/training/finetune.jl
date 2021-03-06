include("../imports.jl")


struct ConstGrouper <: FastAI.ParamGrouper
    g
end
FastAI.group(cg::ConstGrouper, m) = Dict(cg.g => m)

@testset ExtendedTestSet "`finetune!`" begin
    learner = testlearner(Recorder())
    @test_nowarn finetune!(learner, 1; grouper = ConstGrouper(2))

end
