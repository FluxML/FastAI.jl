"""
    TextDatasetRecipe(textfile; labelfn = parentname, split = false)
"""

Base.@kwdef struct TextFolders <: Datasets.DatasetRecipe
    labelfn = parentname
    split::Bool = false
    filefilterfn = _ -> true
end

function Datasets.loadrecipe(recipe::TextFolders, path)
    isdir(path) || error("$path is not a directory")
    data = loadfolderdata(
        path,
        filterfn=f -> istextfile(f) && recipe.filefilterfn(f) && !contains(f, "tmp_clas") && !contains(f, "tmp_lm") && !contains(f, "unsup"),
        loadfn=(loadfile, recipe.labelfn),
        splitfn=recipe.split ? grandparentname : nothing)

    (recipe.split ? length(data) > 0 : nobs(data) > 0) || error("No text files found in $path")

    labels = recipe.split ? first(values(data))[2] : data[2]
    blocks = TextBlock(), Label(unique(eachobs(labels)))
    length(blocks[2].classes) > 1 || error("Expected multiple different labels, got: $(blocks[2].classes))")
    return data, blocks
end

Datasets.recipeblocks(::Type{TextFolders}) = Tuple{TextBlock,Label}

const RECIPES = Dict{String,Vector{Datasets.DatasetRecipe}}(
    "imdb" => [TextFolders()],
)

function _registerrecipes()
    for (name, recipes) in RECIPES, recipe in recipes
        Datasets.registerrecipe!(Datasets.FASTAI_DATA_REGISTRY, name, recipe)
    end
end
    