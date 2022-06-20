Datasets.loadfile(file::String, ::Val{:txt}) = DataFrame(CSV.File(file))

"""
    TableDatasetRecipe(tablefile; catcols, contcols, kwargs...])

Recipe for loading a `TableDataset`. `tablefile` is the path of a file that can
be read as a table. `catcols` and `contcols` indicate the categorical and
continuous columns of the table. If they are not given, they are detected
automatically.
"""
Base.@kwdef struct TableDatasetRecipe <: Datasets.DatasetRecipe
    file = ""
    catcols = nothing
    contcols = nothing
    loadfn = Datasets.loadfile
end

Datasets.recipeblocks(::Type{TableDatasetRecipe}) = TableRow

function Datasets.Datasets.loadrecipe(recipe::TableDatasetRecipe, path)
    tablepath = joinpath(path, recipe.file)
    table = recipe.loadfn(tablepath)
    Tables.istable(table) || error("Expected `recipe.loadfn($(tablepath))` to return a table, instead got type $(typeof(table))")
    data = TableDataset(table)

    catcols, contcols = if isnothing(recipe.catcols) || isnothing(recipe.contcols)
        cat, cont = getcoltypes(data)
        cat = isnothing(recipe.catcols) ? cat : recipe.catcols
        cont = isnothing(recipe.contcols) ? cont : recipe.contcols
        cat, cont
    else
        recipe.catcols, recipe.contcols
    end

    block = TableRow(
        catcols,
        contcols,
        gettransformdict(data, DataAugmentation.Categorify, catcols),
    )
    return data, block
end



struct TableClassificationRecipe <: Datasets.DatasetRecipe
    recipe::TableDatasetRecipe
    targetcol
end

Datasets.recipeblocks(::Type{TableClassificationRecipe}) = (TableRow, Label)


function Datasets.Datasets.loadrecipe(recipe::TableClassificationRecipe, args...; kwargs...)
    data, block::TableRow = Datasets.loadrecipe(recipe.recipe, args...; kwargs...)
    recipe.targetcol in block.catcols || error("Expected categorical column $(recipe.targetcol) to exist.")

    data = rows, labels = (data, Tables.getcolumn(data.table, recipe.targetcol))
    blocks = (
        removecol(block, recipe.targetcol),
        Label(unique(eachobs(labels))),
    )

    return data, blocks
end


struct TableRegressionRecipe <: Datasets.DatasetRecipe
    recipe::TableDatasetRecipe
    targetcol
end

Datasets.recipeblocks(::Type{TableRegressionRecipe}) = (TableRow, Continuous)


function Datasets.Datasets.loadrecipe(recipe::TableRegressionRecipe, args...; kwargs...)
    data, block::TableRow = Datasets.loadrecipe(recipe.recipe, args...; kwargs...)
    recipe.targetcol in block.contcols || error("Expected continuous column $(recipe.targetcol) to exist.")

    data = rows, labels = (data, Tables.getcolumn(data.table, recipe.targetcol))
    blocks = (
        removecol(block, recipe.targetcol),
        Continuous(1),
    )
    return data, blocks
end


# Utils


removecol(block::TableRow, col) = TableRow(
        Tuple(cat for cat in block.catcols if cat != col),
        Tuple(cont for cont in block.contcols if cat != col),
        Dict(catcol => cats for (catcol, cats) in block.categorydict if catcol != col)
    )


# Registering recipes

const RECIPES = Dict{String,Vector{Datasets.DatasetRecipe}}(
    "adult_sample" => [
        TableDatasetRecipe(file="adult.csv"),
        TableClassificationRecipe(TableDatasetRecipe(file="adult.csv"), :salary),
        TableRegressionRecipe(TableDatasetRecipe(file="adult.csv"), :age),
    ],
    "imdb_sample" => [
        TableDatasetRecipe(file="texts.csv"),
        TableClassificationRecipe(TableDatasetRecipe(file="texts.csv"), :label)
    ]
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

@testset "TableDatasetRecipe [recipe]" begin
    path = load(datasets()["adult_sample"])
    recipe = TableDatasetRecipe(file="adult.csv")
    data, block = Datasets.loadrecipe(recipe, path)
    sample = getobs(data, 1)
    @test checkblock(block, sample)
end


@testset "TableClassificationRecipe [recipe]" begin
    path = load(datasets()["adult_sample"])
    recipe = TableClassificationRecipe(TableDatasetRecipe(file="adult.csv"), :salary)
    data, block = Datasets.loadrecipe(recipe, path)
    sample = getobs(data, 1)
    @test checkblock(block, sample)
end


@testset "TableRegressionRecipe [recipe]" begin
    path = load(datasets()["adult_sample"])
    recipe = TableRegressionRecipe(TableDatasetRecipe(file="adult.csv"), :age)
    data, block = Datasets.loadrecipe(recipe, path)
    sample = getobs(data, 1)
    @test checkblock(block, sample)
end
