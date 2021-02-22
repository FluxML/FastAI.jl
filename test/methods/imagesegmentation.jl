include("../imports.jl")


@testset ExtendedTestSet "ImageSegmentation" begin

    method = ImageSegmentation(1:5, (64, 64), downscale = 2)

    DLPipelines.checkmethod_core(method)

end


method = ImageSegmentation(1:5, (64, 64), downscale = 2)
image, mask = DLPipelines.mocksample(method)

encode(method, Training(), (image, mask))
