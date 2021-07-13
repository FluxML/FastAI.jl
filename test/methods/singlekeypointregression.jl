include("../imports.jl")


@testset ExtendedTestSet "SingleKeypointRegression" begin
    method = BlockMethod(
        (Image{2}(), FastAI.Keypoints{2}(10)),
        (
            ProjectiveTransforms((16, 16), inferencefactor=8),
            ImagePreprocessing(),
            ScalePoints((16, 16)),
        )
    )
    DLPipelines.checkmethod_core(method)
    #FastAI.checkmethod_plot(method)
end
