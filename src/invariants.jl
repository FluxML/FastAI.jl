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


"""
    is_data(data)
    is_data(Bool, data)::Bool

Check that `data` implements the data container interface and give detailed info on missing
functionality if not.

Pass `Bool` as a first argument to return a `Bool`.

"""
function is_data(data; kwargs...)
    inv = invariant_datacontainer(; kwargs...)
    return check(inv, data)
end


"""
    is_data(data, block)
    is_data(Bool, data, block)::Bool

Check that `data` implements the data container interface and its observations are valid
instances of `block`, giving detailed errors if not.

Pass `Bool` as a first argument to return a `Bool`.
"""
function is_data(data, block; kwargs...)
    inv = invariant_datacontainer_block(block; kwargs...)
    return check(inv, data)
end

is_data(::Type{Bool}, args...; kwargs...) = convert(Bool, is_data(args...; kwargs...))


function invariant_datacontainer(; var = :data)
    invariant(
        "`$var` implements the data container interface",
        [
            __invariant_numobs(; var),
            invariant("`$var` contains at least one observation") do data
                n = numobs(data)
                if n <= 0
                    return "Instead, got a data container with $n observations."
                end
            end,
            __invariant_getobs(; var),
        ],
        all;
        description="""A data container stores observations and allows (1) getting
            the number of observation and (2) loading an observation.
            See [the tutorial](/documents/docs/tutorials/data_containers.md) for more
            information.""" |> md)
end

function invariant_datacontainer_block(block;
                                       datavar = "data", blockvar = "data", obsvar = "obs")
    return invariant(
        "`$datavar` is a data container with valid observations for block `$(blockname(block))`",
        [

            invariant_datacontainer(; var = datavar),
            invariant(
                invariant_checkblock(block; blockvar = blockvar, obsvar = obsvar);
                inputfn = data -> getobs(data, 1)
            )
        ],
        all)
end

__invariant_getobs(; var = :data) = invariant(
    "`$var` implements the `getobs` interface",
    [
        Invariants.hasmethod_invariant(Base.getindex, :data, :idx => 1)
        Invariants.hasmethod_invariant(MLUtils.getobs, :data, :idx => 1)
    ],
    any;
    description=Invariants.md("""
        `$var` must provide a way load an observation by implementing **either**
        (1) `Base.getindex($var, idx::Int)` (preferred) or (2) `MLUtils.getobs($var, idx::Int)`
        (if regular indexing is already used and has different semantics).
        """),
    inputfn=data -> (; data),
)

__invariant_numobs(; var = :data) = invariant(
    "`$var` implements the `numobs` interface",
    [
        Invariants.hasmethod_invariant(Base.length, :data)
        Invariants.hasmethod_invariant(MLUtils.numobs, :data)
    ],
    any;
    description=Invariants.md("""
        `$var` must provide a way get the number of observations it contains implementing either
        `Base.length($var)` (preferred) or `MLUtils.numobs($var, idx::Int)`
        """),
    inputfn=data -> (; data),
)


@testset "data container invariants" begin
    @testset "is_data" begin
        @test is_data(Bool, 1:10)
        @test !is_data(Bool, nothing)
        @test is_data(Bool, [1])
        @test !is_data(Bool, [])

        @test is_data(Bool, 1:10, Label(1:10))
        @test !is_data(Bool, 0:10, Label(1:10))
        @test is_data(Bool, [[0, 1]], OneHotLabel{Float32}([1, 2]))

        @test is_data(Bool, [(1, 2)], (Label([1]), Label([2])))
    end
end
