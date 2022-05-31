#=
This file implements functions that allow checking common interfaces in the package using
[Invariants.jl](https://github.com/lorenzoh/Invariants.jl).

=#

#=
`is_block_obs(block, obs` chekcs that a value `obs` is a valid observation for a block
`block`.
=#

"""
    is_block(block, obs)

Check whether `obs` is a valid observation for `block` and give
detailed output.

## Examples

{cell=is_block, show=false resultshow=false}
```julia
using FastAI
```

Basic check with a valid observation:

{cell=is_block}
```julia
FastAI.is_block(LabelMulti([1, 2, 3]), [1])
```

An invalid observation will show an error, detailing why the observatio is not valid for
the block:

{cell=is_block}
```julia
FastAI.is_block(LabelMulti([1, 2, 3]), [2, "invalid:("]))
```

As a tuple of blocks is also a valid block, we can check that too. For example, a sample for
a supervised learning task is usually a tuple `(input, label)`. Using `is_block_obs` to
check observations for tuples of blocks (or nested tuples) details which specific
observations are valid.

{cell=is_block}
```julia
using FastAI.Vision: RGB
FastAI.is_block((
        Image{2}(),             # input block
        Label(["cat", "dog"]),  # target block
    ), (
        rand(RGB, 100, 100),    # valid input
        "mouse",                # invalid label
    ))
```

## Extending

To extend this check to work on a new block type `B`, implement
[`invariant_checkblock`](#)`(::B)`. For help implementing invariants see the documentation
of [Invariants.jl](https://github.com/lorenzoh/Invariants.jl).

If `invariant_checkblock` is not implemented for `B`, it will fall back to checking
[`checkblock`](#) which is correct, but doesn't yield helpful output.
"""
function is_block(block, obs; kwargs...)
    inv = invariant_checkblock(block; kwargs...)
    check(inv, obs)
end

function is_block(::Type{Bool}, block, obs; kwargs...)
    inv = invariant_checkblock(block; kwargs...)
    check(Bool, inv, obs)
end

#=
`is_data(data)` checks that `data` is a valid data container.
=#


function is_data(data; kwargs...)
    inv = invariant_datacontainer(; kwargs...)
    return check(inv, data)
end


function invariant_datacontainer(; symbol = :data)
    invariant([
        __invariant_numobs(; symbol),
        __invariant_getobs(; symbol),
    ], "`$symbol` implements the data container interface.",
    description="""A data container stores a number of observations and allows loading
        individual observations. See
        [the tutorial](/documents/docs/tutorials/data_containers.md) for more
        information.""" |> md)
end

__invariant_getobs(; symbol = :data) = invariant([
        Invariants.hasmethod_invariant(Base.getindex, :data, :idx => 1)
        Invariants.hasmethod_invariant(MLUtils.getobs, :data, :idx => 1)
    ],
    "`$symbol` implements the `getobs` interface",
    :any;
    description=Invariants.md("""
        `$symbol` must provide a way load an observation by implementing **either**
        (1) `Base.getindex($symbol, idx::Int)` (preferred) or (2) `MLUtils.getobs($symbol, idx::Int)`
        (if regular indexing is already used and has different semantics).
        """),
    inputfn=data -> (; data),
)

__invariant_numobs(; symbol = :data) = invariant([
        Invariants.hasmethod_invariant(Base.length, :data)
        Invariants.hasmethod_invariant(MLUtils.numobs, :data)
        invariant("`$symbol` is not empty") do input
            n = numobs(input.data)
            if n <= 0
                return "Instead, got a data container with $n observations."
            end
        end
    ],
    "`$symbol` implements the `numobs` interface",
    :any;
    description=Invariants.md("""
        `$symbol` must provide a way get the number of observations it contains implementing either
        `Base.length($symbol)` (preferred) or `MLUtils.numobs($symbol, idx::Int)`
        It must also be non-empty, i.e. contain at least one observation.
        """),
    inputfn=data -> (; data),
)
