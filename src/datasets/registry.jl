
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
    finddatasets(; name, blocks)

Find preconfigured dataset recipes for datasets that match block
types `blocks` in all data sources (if `name == nothing`) or dataset source
with `name`.

Return a vector of `Pair`s `datasetname => recipe`

#### Examples

```julia
TBlocks = Tuple{Image, Label}
datasetname, recipe = finddatasets(blocks=(Image, Label))[1]
data, blocks = recipe(datasetpath(datasetname))
```
"""
function finddatasets(reg::DatasetRegistry; name=nothing, blocks=Any)
    results = collect(Iterators.flatten(((k => v) for v in vs) for (k, vs) in reg.recipes))

    results = filter(((d, recipe),) -> recipeblocks(recipe) <: typify(blocks), results)
    if !isnothing(name)
        results = filter(((d, r),) -> d == name, results)
    end
    return results
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


## Utilities

typify(T::Type) = T
typify(t::Tuple) = Tuple{map(typify, t)...}
typify(block::FastAI.AbstractBlock) = typeof(block)
