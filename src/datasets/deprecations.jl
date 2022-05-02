

"""
    finddatasets(name=nothing, blocks=Any)

Find preconfigured dataset recipes for datasets that match block
types `blocks` in all data sources (if `name == nothing`) or dataset source
with `name`. `blocks` can be given as a type or a nested tuple of block types.

!!! warning "Deprecated"

    This function is deprecated and will be removed in a future version
    of FastAI.jl. Use `filter(datarecipes(); blocks, id = name)` instead.

Return a vector of `Pair`s `datasetname => recipe`

#### Examples

Loading a result

```julia
datasetname, recipe = finddatasets(blocks=(Image, Label))[1]
data, blocks = loadrecipe(recipe, datasetpath(datasetname))
```

Example searches

```
# Single-label image classification
finddatasets(blocks=(Image, Label))

# Single-label classification from any data
finddatasets(blocks=(Any, Label))

# Datasets with images as input data
finddatasets(blocks=(Image, Any))

# All ways to load `pascal2007`
finddatasets(name="pascal2007")
```
"""
finddatasets(; name = "", blocks = Any) = datarecipes(; blocks, id = name)

"""
    loaddataset(name[, blocks = Any]) -> (data, blocks)

Load dataset `name` with a recipe that matches block types
`blocks`. The first matching recipe is selected and loaded.

!!! warning "Deprecated"

    This function is deprecated and will be removed in a future version
    of FastAI.jl. Use `findfirst(datarecipes(); name, blocks)` instead.

## Examples

Load a data container suitable for single-label image classification:
```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
```

Load dataset with any recipe:
```julia
data, blocks = loaddataset(name)
```
"""
loaddataset(name::String, blocks = Any) = load(findfirst(datarecipes(); id = name, blocks))

"""
    listdatasources()

!!! warning "Deprecated"

    This function is deprecated and will be removed in a future version
    of FastAI.jl.

List the dataset sources registered in `registry` (defaults to
`FastAI.defaultdataregistry()`).
"""
listdatasources() = getfield(datasets(), :data).id


"""
    datasetpath(name)

(Down)load registered dataset source named `name` from the dataset registry.
Use [`listdatasources`](#) for a list of all dataset sources.
"""
datasetpath(name::String) = load(datasets()["fastai/$name"])
