

"""
    abstract type AbstractBlockMethod <: LearningTask

Abstract supertype for learning tasks that derive their
functionality from [`Block`](#)s and [`Encoding`](#)s.

These learning tasks require you only to specify blocks and
encodings by defining which blocks of data show up at which
stage of the pipeline. Generally, a subtype will have a field
`blocks` of type `NamedTuple` that contains this information
and a field `encodings` of encodings that are applied to samples.
They can be accessed with `getblocks` and `getencodings`
respectively. For example, [`SupervisedMethod`](#) represents a
learning task where each sample consists of an input and a target.

{cell=main}
```julia
task = SupervisedMethod(
    (Image{2}(), Label(["cat", "dog"])),
    (ImagePreprocessing(), OneHot(),)
)
getblocks(task)
```

To implement a new `AbstractBlockMethod` either

- use the helper [`BlockMethod`](#) (simpler)
- or subtype [`AbstractBlockMethod`](#) (allows customization through
    dispatch)

## Blocks and interfaces

To support different learning task interfaces, a `AbstractBlockMethod`'s
blocks need to contain different blocks. Below we list first block names
with descriptions, and afterwards relevant interface functions and which
blocks are required to use them.

### Blocks

Each name corresponds to a key of the named tuple
`blocks = getblocks(task)`). A block is referred to with `blocks.\$name`
and an instance of data from a block is referred to as `\$name`.

- `blocks.sample`: The most important block, representing one full
    observation of unprocessed data. Data containers used with a learning
    task should have compatible observations, i.e.
    `checkblock(blocks.sample, getobs(data, i))`.
- `blocks.x`: Data that will be fed into the model, i.e. (neglecting batching)
    `model(x)` should work
- `blocks.ŷ`: Data that is output by the model, i.e. (neglecting batching)
    `checkblock(blocks.ŷ, model(x))`
- `blocks.y`: Data that is compared to the model output using a loss function,
    i.e. `lossfn(ŷ, y)`
- `blocks.encodedsample`: An encoded version of `blocks.sample`. Will usually
    correspond to `encodedblockfilled(getencodings(task), blocks.sample)`.

### Interfaces/functionality and required blocks:

Core:
- [`encode`](#)`(task, ctx, sample)` requires `sample`. Also enables use of
    [`taskdataset`](#), [`taskdataloaders`](#)
- [`decode`](#)`(task, ctx, encodedsample)` requires `encodedsample`
- [`decodeypred`](#)`(task, ctx, ŷ)` requires `ŷ`
- [`decodey`](#)`(task, ctx, y)` requires `y`

Training:
- [`taskmodel`](#)`(task)` requires `x`, `ŷ`
- [`tasklossfn`](#)`(task)` requires `y`, `ŷ`

Visualization:
- [`showsample`](#), [`showsamples`](#) require `sample`
- [`showencodedsample`](#), [`showencodedsamples`](#), [`showbatch`](#)
    require `encodedsample`
- [`showsample`](#), [`showsamples`](#) require `sample`
- [`showoutput`](#), [`showoutputs`](#), [`showoutputbatch`](#) require
    `ŷ`, `encodedsample`

Testing:
- [`mockmodel`](#)`(task)` requires `x`, `ŷ`
- [`mocksample`](#)`(task)` requires `sample`


"""
abstract type AbstractBlockMethod <: LearningTask end

getblocks(task::AbstractBlockMethod) = task.blocks
getencodings(task::AbstractBlockMethod) = task.encodings

# Core interface

function encodesample(task::AbstractBlockMethod, context, sample)
    encode(getencodings(task), context, getblocks(task).sample, sample)
end

function encodeinput(task::AbstractBlockMethod, context, input)
    encode(getencodings(task), context, getblocks(task).input, input)
end

function encodetarget(task::AbstractBlockMethod, context, target)
    encode(getencodings(task), context, getblocks(task).target, target)
end

function decode(task::AbstractBlockMethod, context, encodedsample)
    decode(getencodings(task), context, getblocks(task).encodedsample, encodedsample)
end

function decodeypred(task::AbstractBlockMethod, context, ŷ)
    decode(getencodings(task), context, getblocks(task).ŷ, ŷ)
end

function decodey(task::AbstractBlockMethod, context, y)
    decode(getencodings(task), context, getblocks(task).y, y)
end

# Training interface

function taskmodel(task::AbstractBlockMethod, backbone)
    return blockmodel(getblocks(task).x, getblocks(task).ŷ, backbone)
end

function taskmodel(task::AbstractBlockMethod)
    backbone = blockbackbone(getblocks(task).x)
    return blockmodel(getblocks(task).x, getblocks(task).ŷ, backbone)
end

function tasklossfn(task::AbstractBlockMethod)
    return blocklossfn(getblocks(task).ŷ, getblocks(task).y)
end

# Testing interface

mocksample(task::AbstractBlockMethod) = mockblock(task, :sample)
mockblock(task::AbstractBlockMethod, name::Symbol) = mockblock(getblocks(task)[name])

mockmodel(task::AbstractBlockMethod) =
    mockmodel(getblocks(task).x, getblocks(task).ŷ)

"""
    mockmodel(xblock, ŷblock)
    mockmodel(task::AbstractBlockMethod)

Create a fake model that maps batches of block `xblock` to batches of block
`ŷblock`. Useful for testing.
"""
function mockmodel(xblock, ŷblock)
    return function mockmodel_block(xs)
        out = mockblock(ŷblock)
        bs = DataLoaders._batchsize(xs, DataLoaders.BatchDimLast())
        return DataLoaders.collate([out for _ in 1:bs])
    end
end

# ## Block task
#
# `BlockMethod` is a helper to create anonymous block tasks.

"""
    BlockMethod(blocks, encodings)

Create an [`AbstractBlockMethod`](#) directly, passing in a named tuple `blocks`
and `encodings`. See [`SupervisedMethod`](#) for supervised training tasks.
"""
struct BlockMethod{B<:NamedTuple,E} <: AbstractBlockMethod
    blocks::B
    encodings::E
end

Base.show(io::IO, task::BlockMethod) = print(io,
    "BlockMethod(blocks=", keys(getblocks(task)), ")")


# ## Supervised learning task

"""
    SupervisedMethod((inputblock, targetblock), encodings)

A [`AbstractBlockMethod`](#) learning task for the supervised
task of learning to predict a `target` given an `input`. `encodings`
are applied to samples before being input to the model. Model outputs
are decoded using those same encodings to get a target prediction.

In addition, to the blocks defined by [`AbstractBlockMethod`](#),
`getblocks(::SupervisedMethod)` defines the following blocks:

By default the model output is assumed to be an encoded target, but the
`ŷblock` keyword argument to overwrite this.

- `blocks.input`: An unencoded input and the first element in the tuple
    `sample = (input, target)`
- `blocks.target`: An unencoded target and the second element in the tuple
    `sample = (input, target)`
- `blocks.pred`: A prediction. Usually the same as `blocks.target` but may
    differ if a custom `ŷblock` is specified.

A `SupervisedMethod` also enables some additional functionality:

- [`encodeinput`](#)
- [`encodetarget`](#)
- [`showprediction`](#), [`showpredictions`](#)
"""
struct SupervisedMethod{B<:NamedTuple,E} <: AbstractBlockMethod
    blocks::B
    encodings::E
end


function SupervisedMethod(blocks::Tuple{Any,Any}, encodings; ŷblock = nothing)
    sample = input, target = blocks
    x, y = encodedsample = encodedblockfilled(encodings, sample)
    ŷ = isnothing(ŷblock) ? y : ŷblock
    pred = decodedblockfilled(encodings, ŷ)
    blocks = (; input, target, sample, encodedsample, x, y, ŷ, pred)
    SupervisedMethod(blocks, encodings)
end


function Base.show(io::IO, task::SupervisedMethod)
    print(
        io,
        "SupervisedMethod(",
        summary(getblocks(task).input),
        " -> ",
        summary(getblocks(task).target),
        ")",
    )
end


# ## Deprecations

Base.@deprecate BlockMethod(blocks::Tuple{Any, Any}, encodings; kwargs...) SupervisedMethod(blocks, encodings; kwargs...)
