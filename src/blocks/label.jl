
"""
    Label(classes) <: Block
    setup(LabelMulti, data)

`Block` for a categorical label in a single-class context.
`data` is valid for `Label(classes)` if `data ∈ classes`.

See [`LabelMulti`](#) for the multi-class setting where an observation
can belong to multiple classes.

## Examples

```julia
block = Label(["cat", "dog"])  # an observation can be either "cat" or "dog"
@test FastAI.checkblock(block, "cat")
@test !(FastAI.checkblock(block, "horsey"))
```

You can use `setup` to create a `Label` instance from a data container containing
possible classes:

```julia
targets = ["cat", "dog", "dog", "dog", "cat", "dog"]
block = setup(Label, targets)
@test block ≈ Label(["cat", "dog"])
```
"""
struct Label{T} <: Block
    classes::AbstractVector{T}
end

checkblock(label::Label{T}, obs::T) where {T} = obs ∈ label.classes
mockblock(label::Label{T}) where {T} = rand(label.classes)::T

setup(::Type{Label}, data) = Label(unique(eachobs(data)))


function invariant_checkblock(block::Label; blockvar = "block", obsvar = "obs", kwargs...)
    inv = invariant(
            __inv_checkblock_title(block, blockvar, obsvar)
    ) do obs
        if !(obs ∈ block.classes)
            return "\n" * ("""`$obsvar` should be a valid label, i.e. one of
                `$blockvar.classes = $(sprint(show, block.classes, context=:limit => true))`.
                Instead, got invalid value `$(sprint(show, obs))`.
            """ |> Invariants.md)
        end
    end
    invariant(inv; kwargs...)
end


"""
    LabelMulti(classes) <: Block
    setup(LabelMulti, data)

`Block` for a categorical label in a multi-class context where multiple
labels can be associated for an input. Each label must be in `classes`.
For example, for a block `LabelMulti([1, 2, 3])`, `[1, 2]` is a valid
observation, unlike `[0, 2]` (invalid label) or `1` (not a vector of
labels).

Use [`is_block_obs`](#) to make sure you have valid observations.

## Examples

An observation can contain all or none of the listed classes:

```julia
block = LabelMulti(["cat", "dog", "person"])
@test FastAI.checkblock(block, ["cat", "person"])
@test !(FastAI.checkblock(block, ["horsey"]))
```

You can use `setup` to create a `Label` instance from a data container
containing possible classes:

```julia
targets = ["cat", "dog", "dog", "dog", "cat", "dog"]
block = setup(Label, targets)
@test block ≈ Label(["cat", "dog"])
```
"""
struct LabelMulti{T} <: Block
    classes::AbstractVector{T}
end

function checkblock(label::LabelMulti{T}, v::AbstractVector{T}) where {T}
    return all(map(x -> x ∈ label.classes, v))
end

mockblock(label::LabelMulti) =
    unique([rand(label.classes) for _ = 1:rand(1:length(label.classes))])


setup(::Type{LabelMulti}, data) = LabelMulti(unique(eachobs(data)))


Base.summary(io::IO, ::LabelMulti{T}) where {T} = print(io, "LabelMulti{", T, "}")



function invariant_checkblock(block::LabelMulti; blockvar = "block", obsvar = "obs", kwargs...)
    return invariant(
        __inv_checkblock_title(block, blockvar, obsvar),
        [
            invariant("`$obsvar` is an `AbstractVector`",
                description = md("`$obsvar` should be of type `AbstractVector`.")) do obs
                if !(obs isa AbstractVector)
                    return md("Instead, got invalid type `$(typeof(obs))`.")
                end
            end,
            invariant("All elements are valid labels") do obs
                valid = ∈(block.classes).(obs)
                if !(all(valid))
                    unknown = unique(obs[valid .== false])
                    return md("""`$obsvar` should contain only valid labels,
                    i.e. `∀ y ∈ $obsvar: y ∈ $blockvar.classes`, but `$obsvar` includes
                    unknown labels: `$(sprint(show, unknown))`.

                    Valid classes are:
                    `$(sprint(show, block.classes, context=:limit => true))`""")
                end
            end
        ]; kwargs...
    )
end


# ## Tests

@testset "Label [block]" begin
    block = Label(["cat", "dog"])
    @test FastAI.checkblock(block, "cat")
    @test !(FastAI.checkblock(block, "horsey"))

    targets = ["cat", "dog", "dog", "dog", "cat", "dog"]
    block = setup(Label, targets)
    @test block.classes == ["cat", "dog"]

    inv = invariant_checkblock(Label([1, 2, 3]))
    @test check(Bool, inv, 1)
    @test !(check(Bool, inv, 0))
end


@testset "LabelMulti [block]" begin
    block = LabelMulti(["cat", "dog"])
    @test FastAI.checkblock(block, ["cat"])
    @test !(FastAI.checkblock(block, ["horsey", "cat"]))

    targets = ["cat", "dog", "dog", "dog", "cat", "dog"]
    block = setup(LabelMulti, targets)
    @test block.classes == ["cat", "dog"]

    inv = invariant_checkblock(block)
    @test_nowarn check(Exception,  inv, ["cat", "dog"])
    @test check(Bool, inv, [])
    @test !(check(Bool, inv, "cat"))
    @test !(check(Bool, inv, ["mouse"]))
    @test !(check(Bool, inv, ["mouse", "cat"]))
end
