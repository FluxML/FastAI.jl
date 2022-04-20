
function _datareciperegistry(datasetregistry; name = "Dataset recipes")
    Registry(
        name,
        (;
            id = Field(
                String,
                "ID"),
            datasetid = Field(
                String,
                "Dataset ID",
                description = "ID of the dataset this recipe is based on.",
                computefn = function (row, key)
                    val = row[key]
                    if !haskey(datasetregistry, val)
                        throw(ArgumentError("Could not find dataset with ID \"$val\"!"))
                    end
                    return val
                end,
            ),
            blocks = Field(
                Any,
                "Block types",
                description = "Types of blocks of the data container that this recipe loads.",
                getfilterfn = filterblocks,
                formatfn = _formatblock),
            description = Field(
                String,
                "Description",
                optional = true,
                description = "More information about the dataset recipe",
                formatfn=x -> ismissing(x) ? x : Markdown.parse(x)),
            downloaded = Field(
                Bool,
                "Is downloaded",
                description = """
                    Whether the dataset this recipe is based has been downloaded and is
                    available locally.
                """,
                computefn = (row, key) -> isavailable(datasetregistry[row.datasetid].loader)),
            package = Field(
                Module,
                "Package"),
            recipe = Field(
                Datasets.DatasetRecipe,
                "Recipe",
                formatfn = x -> "$(typeof(x).name.name)(...)"
            )
        );
        loadfn = function loadrecipeentry(row)
            dataset = load(datasetregistry[row.datasetid])
            Datasets.loadrecipe(row.recipe, dataset)
        end
    )
end

const DATARECIPES = _datareciperegistry(DATASETS)
datarecipes(; kwargs...) = isempty(kwargs) ? DATARECIPES : find(DATARECIPES; kwargs...)
