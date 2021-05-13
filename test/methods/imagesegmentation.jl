include("../imports.jl")


@testset ExtendedTestSet "ImageSegmentation" begin
    method = ImageSegmentation(1:5, (64, 64), downscale = 1)
    DLPipelines.checkmethod_core(method)
    FastAI.checkmethod_plot(method)
end
