include("../imports.jl")


@testset "ImagePreprocessing" begin
    encfns = [
        () -> ImagePreprocessing(),
        () -> ImagePreprocessing(buffered=false),
        () -> ImagePreprocessing(T=Float64),
        () -> ImagePreprocessing(augmentations=FastAI.augs_lighting()),
        () -> ImagePreprocessing(C=Gray{N0f8}, means=SVector(0.), stds=SVector(1.)),
    ]
    for encfn in encfns
        enc = encfn()
        block = Image{2}()
        img = rand(RGB{N0f8}, 10, 10)
        testencoding(enc, block, img)

        ctx = Validation()
        outblock = encodedblock(enc, block)
        a = encode(enc, ctx, block, img)
        rimg = decode(enc, ctx, outblock, a)
        if eltype(rimg) <: RGB
            @test img ≈ rimg
        end
    end

    @testset "3D" begin
        enc = ImagePreprocessing()
        block = Image{3}()
        img = rand(RGB{N0f8}, 10, 10, 10)
        testencoding(enc, block, img)

        enc = ImagePreprocessing(buffered=false)
        block = Image{3}()
        img = rand(RGB{N0f8}, 10, 10, 10)
        testencoding(enc, block, img)
    end

    @testset ExtendedTestSet "imagedatasetstats" begin

        @testset ExtendedTestSet "RGB" begin
            data = [zeros(RGB{Float32}, 10, 10), ones(RGB{Float32}, 10, 10)]
            means, stds = imagedatasetstats(data, RGB{N0f8}; progress=false)
            @test means ≈ [0.5, 0.5, 0.5]
            @test stds ≈ [0., 0., 0.]
        end

        @testset ExtendedTestSet "Gray" begin
            data = [zeros(Gray{Float32}, 10, 10), ones(Gray{Float32}, 10, 10)]
            means, stds = imagedatasetstats(data, Gray{N0f8}; progress=false)
            @test means ≈ [0.5]
            @test stds ≈ [0.]
        end

    end

    @testset ExtendedTestSet "setup" begin
        data = [
            zeros(10, 10),
            ones(10, 10),
        ]
        enc = setup(ImagePreprocessing, Image{2}(), data, C = Gray{N0f8})
        @test enc.stats[1] ≈ [0.5]
        @test enc.stats[2] ≈ [0.]
    end
end
