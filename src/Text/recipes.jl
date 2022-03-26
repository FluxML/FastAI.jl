
"""
    TextDatasetRecipe(tablefile; catcols, contcols, kwargs...])

Recipe for loading a `TextDataset`. `tablefile` is the path of a file that can
be read as a table. `catcols` and `contcols` indicate the categorical and
continuous columns of the text table. If they are not given, they are detected
automatically.
"""
Base.@kwdef struct TextDatasetRecipe <: Datasets.DatasetRecipe
    file = ""
    catcols = nothing
    contcols = nothing
    loadfn = loadfile
end

Datasets.recipeblocks(::Type{TextDatasetRecipe}) = TextRow

function Datasets.loadrecipe(recipe::TextDatasetRecipe, path)
    tablepath = joinpath(path, recipe.file)
    table = recipe.loadfn(tablepath)
    Tables.istable(table) || error("Expected `recipe.loadfn($(tablepath))` to return a table, instead got type $(typeof(table))")
    data = TextDataset(table)

    catcols, contcols = if isnothing(recipe.catcols) || isnothing(recipe.contcols)
        cat, cont = getcoltypes(data)
        cat = isnothing(recipe.catcols) ? cat : recipe.catcols
        cont = isnothing(recipe.contcols) ? cont : recipe.contcols
        cat, cont
    else
        recipe.catcols, recipe.contcols
    end

    block = TextRow(
        catcols,
        contcols,
        gettransformdict(data, DataAugmentation.Categorify, catcols),
    )
    return data, block
end

struct TextClassificationRecipe <: Datasets.DatasetRecipe
    recipe::TextDatasetRecipe
    targetcol
end

Datasets.recipeblocks(::Type{TextClassificationRecipe}) = (TextRow, Label)


function Datasets.loadrecipe(recipe::TextClassificationRecipe, args...; kwargs...)
    data, block::TextRow = loadrecipe(recipe.recipe, args...; kwargs...)
    recipe.targetcol in block.catcols || error("Expected categorical column $(recipe.targetcol) to exist.")

    data = rows, labels = (data, Tables.getcolumn(data.table, recipe.targetcol))
    blocks = (
        removecol(block, recipe.targetcol),
        Label(unique(eachobs(labels))),
    )

    return data, blocks
end

# Utils


removecol(block::TextRow, col) = TextRow(
    Tuple(cat for cat in block.catcols if cat != col),
    Tuple(cont for cont in block.contcols if cat != col),
    Dict(catcol => cats for (catcol, cats) in block.categorydict if catcol != col)
)


# Registering recipes

const RECIPES = Dict{String,Vector{Datasets.DatasetRecipe}}(
    "ag_news_csv" => [
        TableDatasetRecipe(file="train.csv"),
        # TableClassificationRecipe(TableDatasetRecipe(file="train.csv"), :salary),
    ],
    "amazon_review_full_csv" => [
        TableDatasetRecipe(file="train.csv"),
        # TableClassificationRecipe(TableDatasetRecipe(file="train.csv"), :label)
    ]
)



function _registerrecipes()
    for (name, recipes) in RECIPES, recipe in recipes
        Datasets.registerrecipe!(Datasets.FASTAI_DATA_REGISTRY, name, recipe)
    end
end

