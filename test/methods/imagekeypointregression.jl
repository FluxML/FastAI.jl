include("../imports.jl")


@testset ExtendedTestSet "ImageKeypointRegression" begin
    method = ImageKeypointRegression((16, 16), 10)
    DLPipelines.checkmethod_core(method)
    FastAI.test_method_show(method, ShowText(Base.DevNull()))
end
