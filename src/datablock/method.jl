

abstract type AbstractBlockMethod <: LearningMethod end

getblocks(method::AbstractBlockMethod) = method.blocks
getencodings(method::AbstractBlockMethod) = method.encodings

# Core interface

function encode(method::AbstractBlockMethod, context, sample)
    encode(getencodings(method), context, getblocks(method).sample, sample)
end

function encodeinput(method::AbstractBlockMethod, context, input)
    encode(getencodings(method), context, getblocks(method).input, input)
end

function encodetarget(method::AbstractBlockMethod, context, target)
    encode(getencodings(method), context, getblocks(method).target, target)
end

function decode(method::AbstractBlockMethod, context, encodedsample)
    xyblock = encodedblock(getencodings(method), getblocks(method))
    decode(getencodings(method), context, getblocks(method).encodedsample, encodedsample)
end

function decodeŷ(method::AbstractBlockMethod, context, ŷ)
    decode(getencodings(method), context, getblocks(method).ŷ, ŷ)
end

function decodey(method::AbstractBlockMethod, context, y)
    decode(getencodings(method), context, getblocks(method).y, y)
end

# Training interface

function methodmodel(method::AbstractBlockMethod, backbone)
    return blockmodel(getblocks(method).x, getblocks(method).ŷ, backbone)
end

function methodmodel(method::AbstractBlockMethod)
    backbone = blockbackbone(getblocks(method).x)
    return blockmodel(getblocks(method).x, getblocks(method).ŷ, backbone)
end

function methodlossfn(method::AbstractBlockMethod)
    return blocklossfn(getblocks(method).ŷ, getblocks(method).y)
end

# Testing interface

mocksample(method::AbstractBlockMethod) = mockblock(method, :sample)
mockblock(method::AbstractBlockMethod, name::Symbol) = mockblock(getblocks(method)[name])

mockmodel(method::AbstractBlockMethod) = mockmodel(getblocks(method).x, getblocks(method).ŷ)

function mockmodel(xblock, ŷblock)
    return function mockmodel_block(xs)
        out = mockblock(ŷblock)
        DataLoaders.collate([out])
    end
end


# ## Supervised learning method

"""
    SupervisedMethod((inputblock, targetblock), encodings) <: LearningMethod

Learning method for the supervised task of learning to predict a `target`
given an `input`. `encodings` are applied to samples before being input to
the model. Model outputs are decoded using those same encodings to get
a target prediction.
"""
struct SupervisedMethod{B<:NamedTuple,E} <: AbstractBlockMethod
    blocks::B
    encodings::E
end


function SupervisedMethod(blocks::Tuple{Any, Any}, encodings; ŷblock = nothing)
    sample = input, target = blocks
    x, y = encodedsample = encodedblockfilled(encodings, sample)
    ŷ = isnothing(ŷblock) ? y : ŷblock
    pred = decodedblockfilled(encodings, ŷ)
    blocks = (; input, target, sample, encodedsample, x, y, ŷ, pred)
    SupervisedMethod(blocks, encodings)
end


function Base.show(io::IO, method::SupervisedMethod)
    print(
        io,
        "SupervisedMethod(",
        summary(getblocks(method).input),
        " -> ",
        summary(getblocks(method).target),
        ")",
    )
end


# ## Deprecations

BlockMethod(args...; kwargs...) = SupervisedMethod(args...; kwargs...)
Base.@deprecate BlockMethod(blocks, encodings; kwargs...) SupervisedMethod(
    blocks,
    encodings;
    kwargs...,
)
