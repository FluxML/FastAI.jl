include("../imports.jl")


@testset ExtendedTestSet "ImageSegmentation" begin
    @testset ExtendedTestSet "2D" begin
        method = ImageSegmentation((16, 16), 1:4)
        testencoding(method.encodings, method.blocks)
        DLPipelines.checkmethod_core(method)
        @test_nowarn methodlossfn(method)
        @test_nowarn methodmodel(method, Models.xresnet18())
        FastAI.test_method_show(method, ShowText(Base.DevNull()))
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
