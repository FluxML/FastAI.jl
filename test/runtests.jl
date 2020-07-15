using FastAI
using Test

@testset "FastAI.jl" begin

    include("test_metric.jl")
    include("test_learner.jl")

end
