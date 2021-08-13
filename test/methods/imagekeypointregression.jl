include("../imports.jl")


@testset ExtendedTestSet "ImageKeypointRegression" begin
    method = ImageKeypointRegression((16, 16), 10)
    DLPipelines.checkmethod_core(method)
    FastAI.checkmethod_plot(method)
end
