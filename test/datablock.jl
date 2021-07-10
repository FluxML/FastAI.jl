#include("imports.jl")
using FastAI
import FastAI: Block, Encoding, encode, decode, checkblock, encodedblock, decodedblock
using FastAI: Label, LabelMulti, Mask, Image, ImageTensor, testencoding
using FastAI: OneHot
using Test
using StaticArrays
using Images
using FastAI: grabbounds
using Images

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



@testset "Composition" begin
    encodings = (ProjectiveTransforms((5, 5)), ImagePreprocessing(), OneHot())
    blocks = (Image{2}(), Label(1:10))
    data = (rand(RGB{N0f8}, 10, 10), 7)
    testencoding(encodings, blocks, data)
end


@testset "ProjectiveTransforms" begin
    enc = ProjectiveTransforms((32, 32), buffered = false)
    block = Image{2}()
    image = rand(RGB, 100, 50)

    testencoding(enc, block, image)
    @testset "randstate is shared" begin
        im1, im2 = encode(enc, Training(), (block, block), (image, image))
        im3 = encode(enc, Training(), block, image)
        @test im1 ≈ im2
        @test !(im1 == im3)
    end

    @testset "don't transform data that doesn't need to be resized" begin
        imagesmall = rand(RGB, 32, 32)
        @test imagesmall ≈ encode(enc, Validation(), block, imagesmall)
    end

    @testset "3D" begin

        testencoding(ProjectiveTransforms((16, 16, 16)), Image{3}(), rand(RGB{N0f8}, 32, 24, 24))
    end
end
