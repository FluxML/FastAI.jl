

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

checkblock(label::Label{T}, data::T) where T = data ∈ label.classes
mockblock(label::Label{T}) where T = rand(label.classes)::T

setup(::Type{Label}, data) = Label(unique(eachobs(data)))


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

function checkblock(label::LabelMulti{T}, v::AbstractVector{T}) where T
    return all(map(x -> x ∈ label.classes, v))
end

mockblock(label::LabelMulti) =
    unique([rand(label.classes) for _ in 1:rand(1:length(label.classes))])


setup(::Type{LabelMulti}, data) = LabelMulti(unique(eachobs(data)))


InlineTest.@testset "Label [block]" begin
    block = Label(["cat", "dog"])
    InlineTest.@test FastAI.checkblock(block, "cat")
    InlineTest.@test !(FastAI.checkblock(block, "horsey"))

    targets = ["cat", "dog", "dog", "dog", "cat", "dog"]
    block = setup(Label, targets)
    InlineTest.@test block.classes == ["cat", "dog"]
end


InlineTest.@testset "LabelMulti [block]" begin
    block = LabelMulti(["cat", "dog"])
    InlineTest.@test FastAI.checkblock(block, ["cat"])
    InlineTest.@test !(FastAI.checkblock(block, ["horsey", "cat"]))

    targets = ["cat", "dog", "dog", "dog", "cat", "dog"]
    block = setup(LabelMulti, targets)
    InlineTest.@test block.classes == ["cat", "dog"]
end
