include("../imports.jl")


@testset ExtendedTestSet "ImagePreprocessing" begin
    step = ImagePreprocessing((0, 0, 0), (.5, .5, .5))
    image = rand(RGB, 100, 100)
    x = FastAI.run(step, Training(), image)
    @test size(x) == (100, 100, 3)
    @test eltype(x) == Float32

    image2 = rand(RGB, 100, 100)
    buf = copy(x)
    @test_nowarn FastAI.run!(buf, step, Training(), image2)
    @test !(buf ≈ x)
end
