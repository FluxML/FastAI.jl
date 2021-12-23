

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

Base.summary(io::IO, ::Label{T}) where {T} = print(io, "Label{", T, "}")

function invariant_checkblock(block::Label; blockname = "block", obsname = "obs")
    return BooleanInvariant(
        obs -> obs ∈ block.classes,
        name = "`$obsname` should be a valid `$(summary(block))`",
        messagefn = obs -> """`$obsname` should be one of the valid classes, i.e.
            `$obsname ∈ $blockname.classes`. Instead got unknown class `$(sprint(show, obs))`.
            Valid classes are:
            `$(sprint(show, block.classes, context=:limit => true))`""",
    )
end

"""
    LabelMulti(classes)
    setup(LabelMulti, data)

`Block` for a categorical label in a multi-class context.
`data` is valid for `Label(classes)` if `data ∈ classes`.

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

function invariant_checkblock(block::LabelMulti; blockname = "block", obsname = "obs")
    return SequenceInvariant(
        [
            BooleanInvariant(
                obs -> obs isa AbstractVector,
                name = "`$obsname` should be an `AbstractVector`",
                messagefn = obs -> """`$obsname` should be an `AbstractVector`, instead
                got type `$(typeof(obs))`.
                """
            ),
            BooleanInvariant(
                obs -> all([y ∈ block.classes for y in obs]),
                name = "Elements in `$obsname` should be valid classes",
                messagefn = obs -> """`$obsname` should contain only valid classes,
                    i.e. `∀ y ∈ $obsname: y ∈ $blockname.classes`.
                    Instead got unknown classes `$(
                        sprint(show, [y for y in obs if !(y ∈ block.classes)]))`.

                    Valid classes are:
                    `$(sprint(show, block.classes, context=:limit => true))`""",
            ),
        ],
        "`$obsname` should be a valid `$(summary(block))`",
        "",
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
    @test check(inv, 1)
    @test !(check(inv, 0))
end


@testset "LabelMulti [block]" begin
    block = LabelMulti(["cat", "dog"])
    @test FastAI.checkblock(block, ["cat"])
    @test !(FastAI.checkblock(block, ["horsey", "cat"]))

    targets = ["cat", "dog", "dog", "dog", "cat", "dog"]
    block = setup(LabelMulti, targets)
    @test block.classes == ["cat", "dog"]

    inv = invariant_checkblock(block)
    @test check(inv, ["cat", "dog"])
    @test check(inv, [])
    @test !(check(inv, "cat"))
    @test !(check(inv, ["mouse"]))
    @test !(check(inv, ["mouse", "cat"]))
end
