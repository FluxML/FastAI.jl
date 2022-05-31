

"""
    abstract type AbstractBlockTask <: LearningTask

Abstract supertype for learning tasks that derive their
functionality from [`Block`](#)s and [`Encoding`](#)s.

These learning tasks require you only to specify blocks and
encodings by defining which blocks of data show up at which
stage of the pipeline. Generally, a subtype will have a field
`blocks` of type `NamedTuple` that contains this information
and a field `encodings` of encodings that are applied to samples.
They can be accessed with `getblocks` and `getencodings`
respectively. For example, [`SupervisedTask`](#) represents a
learning task where each sample consists of an input and a target.

{cell=main}
```julia
using FastAI
task = SupervisedTask(
    (Image{2}(), Label(["cat", "dog"])),
    (ImagePreprocessing(), OneHot(),)
)
FastAI.getblocks(task)
```

To implement a new `AbstractBlockTask` either

- use the helper [`BlockTask`](#) (simpler)
- or subtype [`AbstractBlockTask`](#) (allows customization through
    dispatch)

## Blocks and interfaces

To support different learning task interfaces, a `AbstractBlockTask`'s
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
    `checkblock(blocks.sample, data[i])`.
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
abstract type AbstractBlockTask <: LearningTask end

getblocks(task::AbstractBlockTask) = task.blocks
getencodings(task::AbstractBlockTask) = task.encodings

# Core interface

function encodesample(task::AbstractBlockTask, context, sample)
    encode(getencodings(task), context, getblocks(task).sample, sample)
end

function encodeinput(task::AbstractBlockTask, context, input)
    encode(getencodings(task), context, getblocks(task).input, input)
end

function encodetarget(task::AbstractBlockTask, context, target)
    encode(getencodings(task), context, getblocks(task).target, target)
end

function decode(task::AbstractBlockTask, context, encodedsample)
    decode(getencodings(task), context, getblocks(task).encodedsample, encodedsample)
end

function decodeypred(task::AbstractBlockTask, context, ŷ)
    decode(getencodings(task), context, getblocks(task).ŷ, ŷ)
end

function decodey(task::AbstractBlockTask, context, y)
    decode(getencodings(task), context, getblocks(task).y, y)
end

# Training interface

function taskmodel(task::AbstractBlockTask, backbone)
    return blockmodel(getblocks(task).x, getblocks(task).ŷ, backbone)
end

function taskmodel(task::AbstractBlockTask)
    backbone = blockbackbone(getblocks(task).x)
    return blockmodel(getblocks(task).x, getblocks(task).ŷ, backbone)
end

function tasklossfn(task::AbstractBlockTask)
    return blocklossfn(getblocks(task).ŷ, getblocks(task).y)
end

# Testing interface

mocksample(task::AbstractBlockTask) = mockblock(task, :sample)
mockblock(task::AbstractBlockTask, name::Symbol) = mockblock(getblocks(task)[name])

mockmodel(task::AbstractBlockTask) =
    mockmodel(getblocks(task).x, getblocks(task).ŷ)

"""
    mockmodel(xblock, ŷblock)
    mockmodel(task::AbstractBlockTask)

Create a fake model that maps batches of block `xblock` to batches of block
`ŷblock`. Useful for testing.
"""
function mockmodel(_, ŷblock)
    return function mockmodel_block(xs)
        return MLUtils.batch([mockblock(ŷblock) for _ in 1:Datasets.batchsize(xs)])
    end
end

# ## Block task
#
# `BlockTask` is a helper to create anonymous block tasks.

"""
    BlockTask(blocks, encodings)

Create an [`AbstractBlockTask`](#) directly, passing in a named tuple `blocks`
and `encodings`. See [`SupervisedTask`](#) for supervised training tasks.
"""
struct BlockTask{B<:NamedTuple,E} <: AbstractBlockTask
    blocks::B
    encodings::E
end

Base.show(io::IO, task::BlockTask) = print(io,
    "BlockTask(blocks=", keys(getblocks(task)), ")")


# ## Supervised learning task

"""
    SupervisedTask((inputblock, targetblock), encodings)

A [`AbstractBlockTask`](#) learning task for the supervised
task of learning to predict a `target` given an `input`. `encodings`
are applied to samples before being input to the model. Model outputs
are decoded using those same encodings to get a target prediction.

In addition, to the blocks defined by [`AbstractBlockTask`](#),
`getblocks(::SupervisedTask)` defines the following blocks:

By default the model output is assumed to be an encoded target, but the
`ŷblock` keyword argument to overwrite this.

- `blocks.input`: An unencoded input and the first element in the tuple
    `sample = (input, target)`
- `blocks.target`: An unencoded target and the second element in the tuple
    `sample = (input, target)`
- `blocks.pred`: A prediction. Usually the same as `blocks.target` but may
    differ if a custom `ŷblock` is specified.

A `SupervisedTask` also enables some additional functionality:

- [`encodeinput`](#)
- [`encodetarget`](#)
- [`showprediction`](#), [`showpredictions`](#)
"""
struct SupervisedTask{B<:NamedTuple,E} <: AbstractBlockTask
    blocks::B
    encodings::E
end


function SupervisedTask(blocks::Tuple{Any,Any}, encodings; ŷblock = nothing)
    sample = input, target = blocks
    x, y = encodedsample = encodedblockfilled(encodings, sample)
    ŷ = isnothing(ŷblock) ? y : ŷblock
    pred = decodedblockfilled(encodings, ŷ)
    blocks = (; input, target, sample, encodedsample, x, y, ŷ, pred)
    SupervisedTask(blocks, encodings)
end


function Base.show(io::IO, task::SupervisedTask)
    print(
        io,
        "SupervisedTask(",
        summary(getblocks(task).input),
        " -> ",
        summary(getblocks(task).target),
        ")",
    )
end


# ## Deprecations

Base.@deprecate BlockTask(blocks::Tuple{Any, Any}, encodings; kwargs...) SupervisedTask(blocks, encodings; kwargs...)
