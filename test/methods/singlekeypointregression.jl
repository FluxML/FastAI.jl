include("../imports.jl")


@testset ExtendedTestSet "SingleKeypointRegression" begin
    method = SingleKeypointRegression((64, 64))
    DLPipelines.checkmethod_core(method)
    FastAI.checkmethod_plot(method)
end
