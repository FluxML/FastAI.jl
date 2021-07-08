#include("imports.jl")
using FastAI
import FastAI: Block, Encoding, encode, decode, checkblock, encodedblock, decodedblock
using FastAI: Label, LabelMulti, Mask, Image, ImageTensor, testencoding
using FastAI: OneHot
using Test
using StaticArrays

##



struct ABlock <: Block
end
checkblock(::ABlock, ::Int) = true

struct BBlock <: Block
end
checkblock(::BBlock, ::String) = true

struct AtoB <: Encoding end

encode(::AtoB, _, ::ABlock, data) = string(data)
decode(::AtoB, _, ::BBlock, data) = parse(Int, data)
encodedblock(::AtoB, ::ABlock) = BBlock()
decodedblock(::AtoB, ::BBlock) = ABlock()


Test.@testset "Encoding API" begin
    enc = AtoB()
    testencoding(enc, ABlock(), 100)
    testencoding(enc, (ABlock(), ABlock()), (100, 100))
    testencoding((enc,), ABlock(), 100)

    Test.@testset "tuple of encodings" begin
        @test encodedblock((enc, enc,), ABlock()) isa BBlock
        @test encodedblock((enc, enc,), BBlock()) isa Nothing
        @test decodedblock((enc, enc,), BBlock()) isa ABlock
        @test decodedblock((enc, enc,), ABlock()) isa Nothing
    end
end


@testset "OneHot" begin
    enc = OneHot()
    testencoding(enc, Label(1:10), 1)
    testencoding(enc, LabelMulti(1:10), [1])
    testencoding(enc, Mask{2}(1:10), rand(1:10, 50, 50))
end

@testset "ImagePreprocessing" begin
    encfns = [
        () -> ImagePreprocessing(),
        () -> ImagePreprocessing(buffered=false),
        () -> ImagePreprocessing(T=Float64),
        () -> ImagePreprocessing(augmentations=FastAI.augs_lighting()),
        # need fixes in DataAugmentation.jl
        # () -> ImagePreprocessing(C=HSV{Float32}, augmentations=FastAI.augs_lighting()),
        # () -> ImagePreprocessing(C=Gray{N0f8}, means=SVector(0.), stds=SVector(1.)),
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
        @test img ≈ rimg
    end
end


@testset "Composition" begin
    encodings = (ImagePreprocessing(), OneHot())
    blocks = (Image{2}(), Label(1:10))
    data = (rand(RGB{N0f8}, 10, 10), 7)
    testencoding(encodings, blocks, data)
end
