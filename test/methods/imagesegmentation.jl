include("../imports.jl")


@testset ExtendedTestSet "ImageSegmentation" begin
    @testset ExtendedTestSet "2D" begin
        method = BlockMethod(
            (Image{2}(), Mask{2}(1:4)),
            (
                ProjectiveTransforms((16, 16), inferencefactor=8),
                ImagePreprocessing(),
                FastAI.OneHot()
            )
        )
        testencoding(method.encodings, method.blocks)
        DLPipelines.checkmethod_core(method)
        @test_nowarn methodlossfn(method)
        @test_nowarn methodmodel(method, Models.xresnet18())
        FastAI.checkmethod_plot(method)
    end
    @testset ExtendedTestSet "3D" begin
        method = BlockMethod(
            (Image{3}(), Mask{3}(1:4)),
            (
                ProjectiveTransforms((16, 16, 16), inferencefactor=8),
                ImagePreprocessing(),
                FastAI.OneHot()
            )
        )
        testencoding(method.encodings, method.blocks)
        DLPipelines.checkmethod_core(method)
        @test_nowarn methodlossfn(method)
    end

end
