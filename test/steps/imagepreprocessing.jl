include("../imports.jl")


@testset ExtendedTestSet "ImagePreprocessing" begin
    ip = ImagePreprocessing((0, 0, 0), (.5, .5, .5))
    image = rand(RGB{N0f8}, 100, 100)
    x = FastAI.run(ip, Validation(), image)
    @test size(x) == (100, 100, 3)
    @test eltype(x) == Float32

    image2 = rand(RGB{N0f8}, 100, 100)
    buf = copy(x)
    @test_nowarn FastAI.run!(x, ip, Training(), image2)
    @test !(buf ≈ x)
end
