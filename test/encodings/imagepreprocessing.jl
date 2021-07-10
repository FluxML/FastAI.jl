
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

        enc = ImagePreprocessing(buffered = false)
        block = Image{3}()
        img = rand(RGB{N0f8}, 10, 10, 10)
        testencoding(enc, block, img)
    end
end
