
function _datasetregistry(; name = "Datasets")
    registry = Registry(
        name,
        (
            id = Field(String, "ID"),
            name = Field(
                String,
                "Name",
                description = "The name of the dataset",
                computefn = (row, key) -> get(row, key, row.id)),
            size = Field(
                String,
                "Size",
                description = "Download size of the dataset",
                optional = true),
            downloaded = Field(
                Bool,
                "Is downloaded",
                description = """
                    Whether the dataset has been downloaded and is available locally.
                    Updates after session restart.
                """,
                computefn = (row, key) -> isavailable(row.loader)),
            tags = Field(
                Vector{String},
                "Tags",
                defaultfn = (row, key) -> String[]),
            description = Field(
                String,
                "Description",
                optional = true,
                description = "More information about the dataset",
                formatfn=x -> ismissing(x) ? x : Markdown.parse(x)),
            loader = Field(
                DatasetLoader,
                "Dataset loader",
                formatfn = x -> "$(typeof(x))(...)",),
            package = Field(
                Module,
                "Package"),
        ),
        loadfn = row -> loaddata(row.loader),
    )
    return registry
end


DATASETS = _datasetregistry()


function _registerdatasets(registry::Registry)
    for config in FastAI.Datasets.DATASETCONFIGS
        id = "fastai/$(config.datadepname)"
        haskey(registry, id) && continue
        push!(registry, (
            id = id,
            name = config.datadepname,
            description = config.description == "" ? missing : config.description,
            loader = DataDepLoader("fastai-$(config.datadepname)"),
            size=config.size === "???" ? missing : config.size,
            package = FastAI,
        ))
    end
end
