"""
abstract type DatasetRecipe

A recipe that contains configuration for loading a data container. Calling it with a path returns a data container and the blocks that each sample is made of.

#### Interface

- `loadrecipe(::DatasetRecipe, args...) -> (data, blocks)`
- `recipeblocks(::Type{DatasetRecipe}) -> TBlocks`

#### Invariants

- `data` must be a data container of samples that are valid `blocks`, i.e. `checkblock(blocks, getobs(data, 1)) == true`
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
recipeblocks(ImageLabelClf) == Tuple{Image{2}, Label}
```
"""
recipeblocks(::R) where {R<:DatasetRecipe} = recipeblocks(R)


# ## Implementations

# ImageClfFolders

"""
    ImageClfFolders(; labelfn = parentname, split = false)

Recipe for loading a single-label image classification dataset
stored in a hierarchical folder format. If `split == true`, split
the data container on the name of the grandparent folder. The label
defaults to the name of the parent folder but a custom function can
be passed as `labelfn`.

```julia
julia> recipeblocks(ImageClassificationFolders)
Tuple{Image{2}, Label}
```
"""
Base.@kwdef struct ImageClassificationFolders <: DatasetRecipe
    labelfn = parentname
    split::Bool = false
end

function loadrecipe(recipe::ImageClassificationFolders, path)
    isdir(path) || error("$path is not a directory")
    data = loadfolderdata(
        path,
        filterfn=isimagefile,
        loadfn=(loadfile, recipe.labelfn),
        splitfn=recipe.split ? grandparentname : nothing)

    (recipe.split ? length(data) > 0 : nobs(data) > 0) || error("No image files found in $path")

    labels = recipe.split ? first(values(data))[2] : data[2]
    blocks = Image{2}(), Label(unique(eachobs(labels)))
    length(blocks[2].classes) > 1 || error("Expected multiple different labels, got: $(blocks[2].classes))")
    return data, blocks
end

recipeblocks(::Type{ImageClassificationFolders}) = Tuple{Image{2}, Label}
