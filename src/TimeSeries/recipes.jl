Base.@kwdef struct TimeSeriesDatasetRecipe <: Datasets.DatasetRecipe
    file
    loadfn = loadfile
end

Datasets.recipeblocks(::Type{TimeSeriesDatasetRecipe}) = Tuple{TimeSeriesRow, Label} 

function Datasets.loadrecipe(recipe::TimeSeriesDatasetRecipe, path)
    path = convert(String, path)
    datasetpath = joinpath(path, recipe.file)
    rows, labels = recipe.loadfn(datasetpath)
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
        TimeSeriesDatasetRecipe(file="Adiac_TRAIN.ts")
    ],
    "ecg5000" => [
        TimeSeriesDatasetRecipe(file="ECG5000_TRAIN.ts")
    ],
    "natops" => [
        TimeSeriesDatasetRecipe(file="NATOPS_TRAIN.ts")
    ],
)

function _registerrecipes()
    for (name, recipes) in RECIPES, recipe in recipes
        Datasets.registerrecipe!(Datasets.FASTAI_DATA_REGISTRY, name, recipe)
    end
end


## Tests

@testset "TimeSeriesDatasetRecipe [recipe]" begin
    path = datasetpath("adiac")
    recipe = TimeSeriesDatasetRecipe(file="Adiac_TRAIN.arff")
    data, block = loadrecipe(recipe, path)
    sample = getobs(data, 1)
    @test checkblock(block, sample)
end