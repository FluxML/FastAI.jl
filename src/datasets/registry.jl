
abstract type AbstractDatasetRegistry end


"""
    DatasetRegistry()

A store for information on datasets and dataset recipes for loading those datasets.
"""
Base.@kwdef struct DatasetRegistry <: AbstractDatasetRegistry
    datasets::Dict{String,Any} = Dict{String,Any}()
    recipes::Dict{String,Vector{DatasetRecipe}} = Dict{String,Vector{DatasetRecipe}}()
end

## Queries

"""
    listdatasources([registry])

List the dataset sources registered in `registry` (defaults to
`FastAI.defaultdataregistry()`).
"""
listdatasources(reg::DatasetRegistry) = collect(keys(reg.datasets))


"""
    datasetpath([registry], name)

(Down)load registered dataset source named `name` from the dataset registry.
Use [`listdatasources`](#) for a list of all dataset sources.
"""
datasetpath(reg::DatasetRegistry, name::String) = reg.datasets[name]()

"""
    finddatasets([registry]; name=nothing, blocks=Any)

Find preconfigured dataset recipes for datasets that match block
types `blocks` in all data sources (if `name == nothing`) or dataset source
with `name`. `blocks` can be given as a type or a nested tuple of block types.

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
function finddatasets(reg::DatasetRegistry; name=nothing, blocks=Any)::Vector{Pair{String,DatasetRecipe}}
    results = collect(Iterators.flatten(((k => v) for v in vs) for (k, vs) in reg.recipes))
    results = filter(((d, recipe),) -> typify(recipeblocks(recipe)) <: typify(blocks), results)
    if !isnothing(name)
        results = filter(((d, r),) -> d == name, results)
    end
    return results
end


"""
    loaddataset(name[, blocks = Any]) -> (data, blocks)

Load dataset `name` with a recipe that matches block types
`blocks`. The first matching recipe is selected and loaded.

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
function loaddataset(reg::DatasetRegistry, name::String, blocks=Any)
    results = finddatasets(reg; name=name, blocks=blocks)
    isempty(results) && error("Could not find a recipe for dataset \"$name\" that matches block types `$blocks`")
    name_, recipe = results[1]
    @assert name == name_
    return loadrecipe(recipe, datasetpath(reg, name))
end

## Registration

"""
    registerdataset!([registry,] name, loadfn)

Register a dataset in `registry::DatasetRegistry` with a `name` and a
function `loadfn()` that downloads it and returns a path.
"""
function registerdataset!(reg::DatasetRegistry, dataset::String, loadfn)
    dataset ∈ keys(reg.datasets) && error(
        "Dataset name $(dataset) is already registered")
    reg.datasets[dataset] = loadfn
    return reg
end


"""
    registerrecipe!([registry,] name, recipe)

Register a recipe in `registry::DatasetRegistry` for dataset `name`.
"""
function registerrecipe!(reg::DatasetRegistry, dataset::String, recipe::DatasetRecipe)
    dataset ∈ keys(reg.datasets) || error("Dataset $dataset not found.")
    recipes = get!(reg.recipes, dataset, DatasetRecipe[])
    push!(recipes, recipe)
    return reg
end

function blocksmatch(recipe::DatasetRecipe, TBlocks)
    @show recipeblocks(recipe), TBlocks
	return recipeblocks(recipe) <: TBlocks
end
