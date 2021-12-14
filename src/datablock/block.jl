"""
    abstract type AbstractBlock

Abstract supertype of all blocks. You should not subtype form this,
but instead from [`Block`](#) or [`WrapperBlock`](#).
"""
abstract type AbstractBlock end


"""
    abstract type Block

A block describes the meaning of a piece of data in the context of a learning task.
For example, for supervised learning tasks, there is an input and a target
and we want to learn to predict targets from inputs. Learning to predict a
cat/dog label from 2D images is a supervised image classification task that can
be represented with the `Block`s `Image{2}()` and `Label(["cat", "dog"])`.

`Block`s are used in virtually every part of the high-level interfaces, from data
processing over model creation to visualization.

## Extending

Consider the following when subtyping `Block`. A block

- Does not hold observation data itself. Instead they are used in conjunction with
  data to annotate it with some meaning.
- If it has any fields, they should be metadata that cannot be
  derived from the data itself and is constant for every sample in
  the dataset. For example `Label` holds all possible classes which
  are constant for the learning problem.

### Interfaces

There are many interfaces that can be implemented for a `Block`. See the docstrings
of each function for more info about how to implement it.

- [`checkblock`](#)`(block, data)`: check whether a piece of data is a valid block
- [`mockblock`](#)`(block)`: randomly generate a piece of data
- [`blocklossfn`](#)`(predblock, yblock)`: loss function for comparing two blocks
- [`blockmodel`](#)`(inblock, outblock[, backbone])`: construct a task-specific model
- [`blockbackbone`](#)`(inblock)`: construct a backbone model that takes in specific data
- [`plotblock!`](#)`(block, data)`: visualize block data

"""
abstract type Block <: AbstractBlock end


"""
    checkblock(block, data)
    checkblock(blocks, datas)

Check whether `data` is compatible with `block`, returning a `Bool`.

## Examples

```julia
checkblock(Image{2}(), rand(RGB, 16, 16)) == true
```

```julia
checkblock(
    (Image{2}(),        Label(["cat", "dog"])),
    (rand(RGB, 16, 16), "cat"                ),
) == true
```

## Extending

An implementation of `checkblock` should be as specific as possible. The
default method returns `false`, so you only need to implement methods for valid types
and return `true`.
"""
checkblock(::Block, data) = false

function checkblock(blocks::Tuple, datas::Tuple)
    @assert length(blocks) == length(datas)
    return all(checkblock(block, data) for (block, data) in zip(blocks, datas))
end


"""
    mockblock(block)
    mockblock(blocks)

Randomly generate an instance of `block`.
"""
mockblock(blocks::Tuple) = map(mockblock, blocks)


"""
    setup(Block, data)

Create an instance of block type `Block` from data container `data`.

## Examples

```julia
setup(Label, ["cat", "dog", "cat"]) == Label(["cat", "dog"])
```

    setup(Encoding, block, data; kwargs...)

Create an encoding using statistics derived from a data container `data`
with observations of block `block`. Used when some arguments of the encoding
are dependent on the dataset. `data` should be the training dataset. Additional
`kwargs` are passed through to the regular constructor of `Encoding`.

## Examples

```julia
(images, labels), blocks = loaddataset("imagenette2-160", (Image, Label))
setup(ImagePreprocessing, Image{2}(), images; buffered = false)
```

```julia
data, block = loaddataset("adult_sample", TableRow)
setup(TabularPreprocessing, block, data)
```
"""
function setup end


# ## Utilities

typify(T::Type) = T
typify(t::Tuple) = Tuple{map(typify, t)...}
typify(block::FastAI.AbstractBlock) = typeof(block)


# Continous
