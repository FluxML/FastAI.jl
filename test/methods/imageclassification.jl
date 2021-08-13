include("../imports.jl")


@testset "ImageClassification" begin
    method = ImageClassificationMulti((16, 16), [1, 2])
    testencoding(method.encodings, method.blocks)
    DLPipelines.checkmethod_core(method)
    @test_nowarn methodlossfn(method)
    @test_nowarn methodmodel(method, Models.xresnet18())

    @testset "`encodeinput`" begin
        image = rand(RGB, 32, 48)

        xtrain = encodeinput(method, Training(), image)
        @test size(xtrain) == (16, 16, 3)
        @test eltype(xtrain) == Float32

        xinference = encodeinput(method, Inference(), image)
        @test size(xinference) == (16, 24, 3)
        @test eltype(xinference) == Float32
    end
    @testset "`encodetarget`" begin
        category = 1
        y = encodetarget(method, Training(), category)
        @test y ≈ [1, 0]
        # depends on buffered interface for `BlockMethod`s and `Encoding`s
        #encodetarget!(y, method, Training(), 2)
        #@test y ≈ [0, 1]
    end
    FastAI.checkmethod_plot(method)
end

@testset ExtendedTestSet "ImageClassificationMulti" begin

    method = ImageClassificationMulti((16, 16), [1, 2])

    testencoding(method.encodings, method.blocks)
    DLPipelines.checkmethod_core(method)
    @test_nowarn methodlossfn(method)
    @test_nowarn methodmodel(method, Models.xresnet18())
    FastAI.checkmethod_plot(method)
end
