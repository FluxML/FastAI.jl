include("../imports.jl")


@testset "keypointpreprocessing.jl" begin

    ks = [
        SVector{2, Float32}(10, 10),
        SVector{2, Float32}(50, 80),
    ]
    sz = (100, 100)

    block = Keypoints{2}(2)
    enc = KeypointPreprocessing(sz)
    ctx = Training()

    testencoding(enc, block, ks)
    y = encode(enc, ctx, block, ks)
    ks_ = decode(enc, ctx, encodedblock(enc, block), y)
    @test ks ≈ ks_

end
