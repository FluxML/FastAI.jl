

"""
    abstract type ShowBackend

Abstract type for backends that allow showing blocks of data in an
interpretable way.

## Extending

For a `ShowBackend` `Backend`, you should implement the following methods:

- [`createhandle`](#)`(::Backend)` creates a context that blocks of data can be shown to
- [`showblock!`](#)`(handle, ::Backend, block::B, data)` shows a block of type `B`. This
    needs to be implemented for every block type you want to be able to show
- [`showblocks!`](#)`(handle, ::Backend, blocks, datas)` shows a collection of blocks
"""
abstract type ShowBackend end


"""
    createhandle(backend::ShowBackend)

Creates a context to which blocks of data can be shown using the
mutating functions `showblock!` and `showblocks!`. It is called internally
when using `showblock` or `showblocks`.

```julia
handle = createhandle(backend)
showblock!(handle, backend, block, data)

# Above is equivalent to
showblock(backend, block, data)
```
"""
function createhandle end



"""
    showblock!(handle, backend, block, data)
    showblock!(handle, backend, blocks, datas)
    showblock!(handle, backend, title => block, data)

Show block of data to an existing context `handle` using `backend`.

See [`showblock`](#) for examples.

## Extending

Every `ShowBackend` should implement the following versions of this method:

- `showblock!(handle, backend, block::Block, data)` to show a single block of data;
    should be implemented for every block type you want to show
- `showblock!(handle, backend, blocks::Tuple, datas::Tuple)` to show several blocks that
    belong to the same observation.

Optionally, you can also implement

- `showblock!(handle, backend, pair::Pair, data)` where `(title, block) = pair` gives
    the name for a block. If this is not implemented for a backend, then calling it
    will default to the untitled method.
"""
function showblock! end

showblock!(handle, backend, (title, block)::Pair, data) =
    showblock!(handle, backend, block, data)


"""
    showblock([backend], block, data)
    showblock([backend], blocks, datas)
    showblock([backend], title => block, data)

Show a block or blocks of data to `backend <: ShowBackend`.

`block` can be a `Block`, a tuple of `block`s, or a `Pair` of `title => block`.
"""
function showblock(backend::ShowBackend, block, data)
    handle = createhandle(backend)
    showblock!(handle, backend, block, data)
end


"""
    showblocks([backend], block, datas)
    showblocks!(handle, backend, block, datas)

Show a vector of observations `datas` of the same `block` type.

## Examples

```julia
data, blocks = loaddataset("imagenette2-160")
samples = [getobs(data, i) for i in range(1:4)]
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
showblocks(backend::ShowBackend, block, datas) =
    showblocks!(createhandle(backend), backend, block, datas)
