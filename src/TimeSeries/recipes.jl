"""
    TimeSeriesDatasetRecipe(file; loadfn = loadfile)

Recipe for loading a time series dataset stored in a .ts file

"""
Base.@kwdef struct TimeSeriesDatasetRecipe <: Datasets.DatasetRecipe
    train_file
    test_file = nothing
    loadfn = Datasets.loadfile
end

Datasets.recipeblocks(::Type{TimeSeriesDatasetRecipe}) = Tuple{TimeSeriesRow, Label} 

#TODO: Add Check if test_file is nothing.
function Datasets.loadrecipe(recipe::TimeSeriesDatasetRecipe, path)
    path = convert(String, path)
    datasetpath_train = joinpath(path, recipe.train_file)
    rows_train, labels_train = recipe.loadfn(datasetpath_train)
    datasetpath_test = joinpath(path, recipe.test_file)
    rows_test, labels_test = recipe.loadfn(datasetpath_test)
    rows = [rows_train; rows_test]
    labels = [labels_train; labels_test]
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
    "ecg5000" => [
        TimeSeriesDatasetRecipe(train_file="ECG5000_TRAIN.ts", test_file="ECG5000_TEST.ts")
    ],
)

function _registerrecipes()
    for (name, recipes) in RECIPES, recipe in recipes
        if !haskey(datarecipes(), name)
            push!(datarecipes(), (
                id = name,
                datasetid = name,
                blocks = Datasets.recipeblocks(recipe),
                package = @__MODULE__,
                recipe = recipe,
            ))
        end
    end
end

# ## Tests

@testset "TimeSeriesDataset [recipe]" begin
    path = datasetpath("ecg5000")
    recipe = TimeSeriesDatasetRecipe(train_file="ECG5000_TRAIN.ts", test_file="ECG5000_TEST.ts")
    data, block = loadrecipe(recipe, path)
    sample = getobs(data, 1)
    @test checkblock(block, sample)
end