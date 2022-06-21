

"""
    abstract type ShowBackend

Abstract type for backends that allow showing blocks of data in an
interpretable way.

## Extending

For a `ShowBackend` `Backend`, you should implement the following methods:

- [`createhandle`](#)`(::Backend)` creates a context that blocks of data can be shown to
- [`showblock!`](#)`(handle, ::Backend, block::B, obs)` shows a block of type `B`. This
    needs to be implemented for every block type you want to be able to show
- [`showblocks!`](#)`(handle, ::Backend, blocks, obss)` shows a collection of blocks
"""
abstract type ShowBackend end


"""
    createhandle(backend::ShowBackend)

Creates a context to which blocks of data can be shown using the
mutating functions `showblock!` and `showblocks!`. It is called internally
when using `showblock` or `showblocks`.

```julia
handle = createhandle(backend)
showblock!(handle, backend, block, obs)

# Above is equivalent to
showblock(backend, block, obs)
```
"""
function createhandle end



"""
    showblock!(handle, backend, block, obs)
    showblock!(handle, backend, blocks, obss)
    showblock!(handle, backend, title => block, obs)

Show block of data to an existing context `handle` using `backend`.

See [`showblock`](#) for examples.

## Extending

Every `ShowBackend` should implement the following versions of this method:

- `showblock!(handle, backend, block::Block, obs)` to show a single block of obs;
    should be implemented for every block type you want to show
- `showblock!(handle, backend, blocks::Tuple, obss::Tuple)` to show several blocks that
    belong to the same observation.

Optionally, you can also implement

- `showblock!(handle, backend, pair::Pair, obs)` where `(title, block) = pair` gives
    the name for a block. If this is not implemented for a backend, then calling it
    will default to the untitled method.
"""
function showblock! end

showblock!(handle, backend, (title, block)::Pair, obs) =
    showblock!(handle, backend, block, obs)


"""
    showblock([backend], block, obs)
    showblock([backend], blocks, obss)
    showblock([backend], title => block, obs)

Show a block or blocks of obs to `backend <: ShowBackend`.

`block` can be a `Block`, a tuple of `block`s, or a `Pair` of `title => block`.
"""
function showblock(backend::ShowBackend, block, obs)
    handle = createhandle(backend)
    showblock!(handle, backend, block, obs)
end


"""
    showblocks([backend], block, obss)
    showblocks!(handle, backend, block, obss)

Show a vector of observations `obss` of the same `block` type.

## Examples

```julia
data, blocks = loaddataset("imagenette2-160")
samples = [data[i] for i in range(1:4)]
showblocks(blocks, samples)
```

## Extending

This is used for showing batches of observations, unlike the `Tuple` variant
of `showblock!` which assumes an observation consists of multiple blocks.

Usually, a [`ShowBackend`](#) will show an observation in one row with `showblock!`
and `showblocks!` will show multiple rows.
"""
function showblocks! end


Base.@doc (Base.@doc showblocks!)
showblocks(backend::ShowBackend, block, obss) =
    showblocks!(createhandle(backend), backend, block, obss)


# WrapperBlock handling

showblock!(handle, backend::ShowBackend, block::WrapperBlock, obs) =
    showblock!(handle, backend, wrapped(block), obs)

isshowable(backend::ShowBackend, wrapper::WrapperBlock) =
    isshowable(backend, wrapped(wrapper))
