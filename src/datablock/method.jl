
"""
    BlockMethod(blocks, encodings) <: LearningMethod

Learning method based on the `Block` and `Encoding` interfaces.
"""
struct BlockMethod{B, E, O} <: LearningMethod
    blocks::B
    encodings::E
    outputblock::O
end

function BlockMethod(blocks, encodings; outputblock = encodedblockfilled(encodings, blocks[2]))
    return BlockMethod(blocks, encodings, outputblock)
end


# Core interface

function encode(method::BlockMethod, context, sample)
    encode(method.encodings, context, method.blocks, sample)
end

function encodeinput(method::BlockMethod, context, input)
    encode(method.encodings, context, method.blocks[1], input)
end

function encodetarget(method::BlockMethod, context, target)
    encode(method.encodings, context, method.blocks[2], target)
end

function decode(method::BlockMethod, context, xy)
    xyblock = encodedblock(method.encodings, method.blocks)
    decode(method.encodings, context, xyblock, xy)
end

function decodeŷ(method::BlockMethod, context, ŷ)
    decode(method.encodings, context, method.outputblock, ŷ)
end

function decodey(method::BlockMethod, context, y)
    yblock = encodedblock(method.encodings, method.blocks[2])
    decode(method.encodings, context, yblock, y)
end

# Training interface

function methodmodel(method::BlockMethod, backbone)
    xblock = encodedblockfilled(method.encodings, method.blocks[1])
    return blockmodel(xblock, method.outputblock, backbone)
end

function methodmodel(method::BlockMethod)
    xblock = encodedblockfilled(method.encodings, method.blocks[1])
    return blockmodel(xblock, method.outputblock, blockbackbone(xblock))
end

function methodlossfn(method::BlockMethod)
    yblock = encodedblockfilled(method.encodings, method.blocks[2])
    return blocklossfn(method.outputblock, yblock)
end

# Testing interface

mocksample(method::BlockMethod) = mockblock(method.blocks)

mockmodel(method::BlockMethod) = mockmodel(
    encodedblock(method.encodings, method.blocks[1]),
    method.outputblock
)

function mockmodel(inblock::AbstractBlock, outblock::AbstractBlock)
    return function mockmodel_block(xs)
        out = mockblock(outblock)
        DataLoaders.collate([out])
    end
end

# Pretty-printing

function Base.show(io::IO, method::BlockMethod)
    print(io, "BlockMethod(", typeof(method.blocks[1]), " -> ", typeof(method.blocks[2]), ")")
end
