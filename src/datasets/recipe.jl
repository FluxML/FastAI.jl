"""
    abstract type DatasetRecipe

A recipe that contains configuration for loading a data container.
Calling it with a path returns a data container and the blocks
that each sample is made of.

## Examples

For example implementations, see [`Vision.ImageFolders`](#).

## Extending

### Interface

- [`loadrecipe`](#)`(::DatasetRecipe, args...; kwargs...) -> (data, blocks)`
    This loads a data container the [`Block`]s that each observation corresponds to.
    For most recipes the only argument beside the recipe is a path to a folder on disk.
- [`recipeblocks`](#)`(::Type{DatasetRecipe}) -> TBlocks`
    The type of `blocks` returned by `loadrecipe`. Should be as specific as possible.
    Used for discovery.

### Invariants

Given

```julia
data, blocks = loadrecipe(recipe, args...; kwargs...)
```

the following must hold:

- `∀i ∈ [1..nobs(data)]: checkblock(blocks, getobs(data, i))`, i.e.
    `data` must be a data container of observations that are valid `blocks`.
- `nobs(data) ≥ 1`, i.e. there is at least one observation if the data was loaded
    without error.
"""
abstract type DatasetRecipe end


"""
    loadrecipe(recipe, path)

Load a recipe from a path. Return a data container `data` and concrete
`blocks`.
"""
function loadrecipe end


"""
    recipeblocks(TRecipe) -> TBlocks
    recipeblocks(recipe) -> TBlocks

Return the `Block` _types_ for the data container that recipe
type `TRecipe` creates. Does not return `Block` instances as the exact
configuration may not be known until the dataset is being
loaded.

#### Examples

```julia
recipeblocks(ImageFolders) == Tuple{Image{2}, Label}
```
"""
recipeblocks(::R) where {R <: DatasetRecipe} = recipeblocks(R)


function testrecipe(recipe::Datasets.DatasetRecipe, path::AbstractPath)
    data, blocks = loadrecipe(recipe, path)
    return testrecipe(recipe, data, blocks)
end

function testrecipe(recipe::Datasets.DatasetRecipe, data, blocks)
    # `blocks` must be an instance of `recipeblocks(recipe)`
    @test blocks isa Datasets.recipeblocks(recipe)

    # Observations must be compatible with `blocks`
    @test checkblock(blocks, getobs(data, 1))
end
