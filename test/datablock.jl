#include("imports.jl")
using FastAI
import FastAI: Block, Encoding, encode, decode, checkblock, encodedblock, decodedblock
using FastAI: Label, LabelMulti, Mask, Image, ImageTensor, testencoding
using FastAI: OneHot
using Test



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
