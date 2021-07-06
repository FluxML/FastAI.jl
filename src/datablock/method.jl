
"""
    BlockMethod(blocks, encodings) <: LearningMethod

Learning method based on the `Block` and `Encoding` interfaces.
"""
struct BlockMethod{B, E, O} <: LearningMethod
    blocks::B
    encodings::E
    outputblock::O
end

function BlockMethod(blocks, encodings; outputblock = encodedblock(encodings, blocks[1]))
    return BlockMethod(blocks, encodings, outputblock)
end


function encode(method::BlockMethod, context, sample)
    encode(method.encodings, context, method.blocks, sample)
end


function decodeŷ(method::BlockMethod, context, ŷ)
    decode(method.encodings, context, method.outputblock, ŷ)
end
