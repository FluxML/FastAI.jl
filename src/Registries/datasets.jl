
function _datasetregistry(; name = "Datasets")
    registry = Registry(
        (
            id = Field(
                String,
                name = "ID",
                formatfn=FeatureRegistries.string_format),
           description = Field(
                String;
                name = "Description",
                optional = true,
                description = "More information about the dataset",
                formatfn=FeatureRegistries.md_format),
            size = Field(
                String;
                name = "Size",
                description = "Download size of the dataset",
                optional = true),
            tags = Field(
                Vector{String};
                name = "Tags",
                defaultfn = (row, key) -> String[]),
            package = Field(
                Module;
                name = "Package",
                formatfn = FeatureRegistries.code_format),
            downloaded = Field(
                Bool;
                name = "Is downloaded",
                description = """
                    Whether the dataset has been downloaded and is available locally.
                    Updates after session restart.
                """,
                computefn = (row, key) -> isavailable(row.loader)),
            loader = Field(
                DatasetLoader;
                name = "Dataset loader",
                formatfn = x -> FeatureRegistries.type_format),
        );
        name,
        loadfn = function (row)
            if row.loader isa DataDepLoader && startswith(row.loader.datadep, "fastai-")
                # Change download format for fastai datasets without having to redownload
                # The download process was slightly changed; this saves having to
                # redownload to update.
                dir = Datasets.loaddata(row.loader)
                if isdir(joinpath(dir, row.id))
                    cd(dir) do
                        temp = mktempdir()
                        mv(joinpath(dir, row.id), temp, force=true)
                        mv(temp, pwd(), force=true)
                    end
                end
            end
            loaddata(row.loader)
        end,
        description = """
        A registry for datasets. `load`ing an entry will download a dataset (if it
        hasn't been already), and return a path to where the files were downloaded.

        ```julia
        path = load(datasets()[id])
        ```

        See `datarecipes` to load these datasets in a format compatible with learning
        tasks.
        """
    )
    return registry
end


const DATASETS = _datasetregistry()



"""
    datasets(; filters...)

Show a registry of available datasets. Pass in filters as keyword
arguments to look at a subset.

See also [finding functionality](/documents/docs/discovery.md), [`learningtasks`](#),
and [`datarecipes`](#). For more information on registries, see
[FeatureRegistries.jl](https://github.com/lorenzoh/FeatureRegistries.jl).


## Examples

Show all available learning tasks:

{cell}
```julia
using FastAI
datasets()
```

Download a dataset:

{cell}
```julia
path = load(datasets()["imagenette2-160"])
```

Get an explanation of fields in the dataset registry:

{cell}
```julia
info(datasets())
```

Show all datasets with `"image"` in their name:

{cell}
```julia
datasets(id="image")
```
"""
datasets(; kwargs...) = isempty(kwargs) ? DATASETS : filter(DATASETS; kwargs...)


function _registerdatasets(registry::Registry)
    for config in FastAI.Datasets.DATASETCONFIGS
        id = config.datadepname
        haskey(registry, id) && continue
        push!(registry, (
            id,
            description = config.description == "" ? missing : config.description,
            loader = DataDepLoader("fastai-$(config.datadepname)"),
            size=config.size === "???" ? missing : config.size,
            package = FastAI,
        ))
    end
end
