"""
    TableDatasetRecipe(tablefile; catcols, contcols, kwargs...])

Recipe for loading a `TableDataset`. `tablefile` is the path of a file that can
be read as a table. `catcols` and `contcols` indicate the categorical and
continuous columns of the table. If they are not given, they are detected
automatically.
"""
Base.@kwdef struct TableDatasetRecipe <: DatasetRecipe
    file = ""
    catcols = nothing
    contcols = nothing
    loadfn = loadfile
end

recipeblocks(::Type{TableDatasetRecipe}) = TableRow

function loadrecipe(recipe::TableDatasetRecipe, path)
    tablepath = joinpath(path, recipe.file)
    table = recipe.loadfn(tablepath)
    Tables.istable(table) || error("Expected `recipe.loadfn($(tablepath))` to return a table, instead got type $(typeof(table))")
    data = TableDataset(table)

    catcols, contcols = if isnothing(recipe.catcols) || isnothing(recipe.contcols)
        cat, cont = FastAI.getcoltypes(data)
        cat = isnothing(recipe.catcols) ? cat : recipe.catcols
        cont = isnothing(recipe.contcols) ? cont : recipe.contcols
        cat, cont
    else
        recipe.catcols, recipe.contcols
    end

    block = TableRow(
        catcols,
        contcols,
        FastAI.gettransformdict(data, DataAugmentation.Categorify, catcols),
    )
    return data, block
end



struct TableClassificationRecipe <: DatasetRecipe
    recipe::TableDatasetRecipe
    targetcol
end

recipeblocks(::Type{TableClassificationRecipe}) = (TableRow, Label)


function loadrecipe(recipe::TableClassificationRecipe, args...; kwargs...)
    data, block::TableRow = loadrecipe(recipe.recipe, args...; kwargs...)
    recipe.targetcol in block.catcols || error("Expected categorical column $(recipe.targetcol) to exist.")

    data = rows, labels = (data, Tables.getcolumn(data.table, recipe.targetcol))
    blocks = (
        removecol(block, recipe.targetcol),
        Label(unique(eachobs(labels))),
    )

    return data, blocks
end


struct TableRegressionRecipe <: DatasetRecipe
    recipe::TableDatasetRecipe
    targetcol
end

recipeblocks(::Type{TableRegressionRecipe}) = (TableRow, Continuous)


function loadrecipe(recipe::TableRegressionRecipe, args...; kwargs...)
    data, block::TableRow = loadrecipe(recipe.recipe, args...; kwargs...)
    recipe.targetcol in block.contcols || error("Expected continuous column $(recipe.targetcol) to exist.")

    data = rows, labels = (data, Tables.getcolumn(data.table, recipe.targetcol))
    blocks = (
        removecol(block, recipe.targetcol),
        Continuous(1),
    )
    return data, blocks
end





## Utils


removecol(block::TableRow, col) = TableRow(
        Tuple(cat for cat in block.catcols if cat != col),
        Tuple(cont for cont in block.contcols if cat != col),
        Dict(catcol => cats for (catcol, cats) in block.categorydict if catcol != col)
    )
