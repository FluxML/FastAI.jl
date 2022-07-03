
function _datareciperegistry(datasetregistry; name = "Dataset recipes")
    Registry((;
              id = Field(String, name = "ID", formatfn = x -> sprint(show, x)),
              blocks = Field(Any,
                             name = "Block types",
                             description = "Types of blocks of the data container that this recipe loads.",
                             filterfn = blocktypesmatch,
                             formatfn = b -> FeatureRegistries.code_format(_formatblock(b))),
              description = Field(String,
                                  name = "Description",
                                  optional = true,
                                  description = "More information about the dataset recipe",
                                  formatfn = FeatureRegistries.md_format),
              downloaded = Field(Bool,
                                 name = "Is downloaded",
                                 description = """
                                     Whether the dataset this recipe is based has been downloaded and is
                                     available locally.
                                 """,
                                 computefn = (row, key) -> isavailable(datasetregistry[row.datasetid].loader)),
              datasetid = Field(String,
                                name = "Dataset ID",
                                description = "ID of the dataset this recipe is based on.",
                                computefn = function (row, key)
                                    val = row[key]
                                    if !haskey(datasetregistry, val)
                                        throw(ArgumentError("Could not find dataset with ID \"$val\"!"))
                                    end
                                    return val
                                end),
              package = Field(Module,
                              name = "Package"),
              recipe = Field(Datasets.DatasetRecipe,
                             name = "Recipe",
                             formatfn = FeatureRegistries.type_format));
             name,
             loadfn = function loadrecipeentry(row)
                 dataset = load(datasetregistry[row.datasetid])
                 Datasets.loadrecipe(row.recipe, dataset)
             end,
             description = """
             A registry for dataset recipes. `load`ing an entry will download the
             corresponding dataset (see `datasets`) and return `data, blocks`, a
             data container and the `Block`s of the observations.

              ```julia
              data, blocks = load(datarecipes()[id])
              ```

             See `learningtasks` to find compatible learning tasks.
             """)
end

const DATARECIPES = _datareciperegistry(DATASETS)

"""
    datarecipes(; filters...)

Show a registry of available dataset recipes. A dataset recipe defines how
to load a dataset into a suitable format for use with a learning task.

Pass in filters as keyword arguments to look at a subset.

See also [finding functionality](/documents/docs/discovery.md), [`datasets`](#),
and [`learningtasks`](#). For more information on registries, see
[FeatureRegistries.jl](https://github.com/lorenzoh/FeatureRegistries.jl).

## Examples

Show all available dataset recipes:

{cell}
```julia
using FastAI
datarecipes()
```

Show all recipes for datasets that have "image" in their name:

{cell}
```julia
datarecipes(datasetid="image")
```

Show all data recipes usable for classification tasks, that is where the
target block is a [`Label`](#):

{cell}
```julia
datarecipes(blocks=(Any, Label))
```

Get an explanation of fields in the dataset recipe registry:

{cell}
```julia
info(datarecipes())
```
"""
datarecipes(; kwargs...) = isempty(kwargs) ? DATARECIPES : filter(DATARECIPES; kwargs...)

registerrecipes(m::Module, recipes) = registerrecipes(DATARECIPES, m, recipes)

function registerrecipes(reg::Registry, m::Module, recipes::Dict)
    for (datasetid, rs) in recipes, recipe in rs
        recipeid, recipe = if recipe isa Pair
            joinpath(datasetid, recipe[1]), recipe[2]
        else
            datasetid, recipe
        end

        if !haskey(reg, recipeid)
            push!(reg,
                  (id = recipeid,
                   datasetid = datasetid,
                   blocks = Datasets.recipeblocks(recipe),
                   package = m,
                   recipe = recipe))
        end
    end
end
