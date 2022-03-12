Base.@kwdef struct TimeSeriesDatasetRecipe <: Datasets.DatasetRecipe
    file
    targetcol = "target"
end

Datasets.recipeblocks(::Type{TimeSeriesDatasetRecipe}) = Tuple{TimeSeriesRow, Label} 

function Datasets.loadrecipe(recipe::TimeSeriesDatasetRecipe, path)
    path = convert(String, path)
    datasetpath = joinpath(path, recipe.file)
    df = ARFFFiles.load(DataFrame, datasetpath)
    labels = Array(df[!, recipe.targetcol])
    rows = Matrix(select(df, Not(:target)))
    N,M = size(rows)
    rows = reshape(rows, (N,1,M))
    rows = TimeSeriesDataset(rows)
    data = rows, labels
    blocks = (
        setup(TimeSeriesRow,rows),
        Label(unique(eachobs(labels))),
    )

    return data, blocks
end

# Registering recipes

const RECIPES = Dict{String,Vector{Datasets.DatasetRecipe}}(
    "adiac" => [
        TimeSeriesDatasetRecipe(file="Adiac_TRAIN.arff")
    ]
)

function _registerrecipes()
    for (name, recipes) in RECIPES, recipe in recipes
        Datasets.registerrecipe!(Datasets.FASTAI_DATA_REGISTRY, name, recipe)
    end
end


# ## Tests


@testset "TimeSeriesDatasetRecipe [recipe]" begin
    path = datasetpath("adiac")
    recipe = TimeSeriesDatasetRecipe(file="Adiac_TRAIN.arff")
    data, block = loadrecipe(recipe, path)
    sample = getobs(data, 1)
    @test checkblock(block, sample)
end