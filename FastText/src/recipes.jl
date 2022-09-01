Datasets.loadfile(file::String, ::Val{:txt}) = read(file, String)

const RE_TEXTFILE = r".*\.(txt|csv|json|md|html?|xml|yaml|toml)$"i
istextfile(f) = matches(RE_TEXTFILE, f)

"""
    TextFolders(textfile; labelfn = parentname, split = false)

Recipe for loading a single-label text classification dataset
stored in hierarchical folder format.
"""
Base.@kwdef struct TextFolders <: Datasets.DatasetRecipe
    labelfn = Datasets.parentname
    split::Bool = false
    filefilterfn = _ -> true
end

Datasets.recipeblocks(::Type{TextFolders}) = Tuple{Paragraph,Label}

function Datasets.loadrecipe(recipe::TextFolders, path)
    isdir(path) || error("$path is not a directory")
    data = loadfolderdata(path,
        filterfn = f -> istextfile(f) && recipe.filefilterfn(f),
        loadfn = (loadfile, recipe.labelfn),
        splitfn = recipe.split ? grandparentname : nothing)

    (recipe.split ? length(data) > 0 : numobs(data) > 0) ||
        error("No text files found in $path")

    labels = recipe.split ? first(values(data))[2] : data[2]
    blocks = (Paragraph(), Label(unique(eachobs(labels))))
    length(blocks[2].classes) > 1 ||
        error("Expected multiple different labels, got: $(blocks[2].classes))")
    return data, blocks
end

Base.@kwdef struct TextGenerationFolders <: Datasets.DatasetRecipe
    textgenerationfolder::String = "unsup"
    labelfn = Datasets.parentname
    split::Bool = false
    filefilterfn = _ -> true
end

Datasets.recipeblocks(::Type{TextGenerationFolders}) = Tuple{Paragraph}

function Datasets.loadrecipe(recipe::TextGenerationFolders, path)
    isdir(path) || error("$path is not a directory")
    textpath = joinpath(path, recipe.textgenerationfolder)
    isdir(textpath) || error("$textpath is not a directory")
    data = loadfolderdata(textpath,
        filterfn = f -> istextfile(f) && recipe.filefilterfn(f),
        loadfn = (loadfile, recipe.labelfn),
        splitfn = recipe.split ? parentname : nothing)
    labels = recipe.split ? first(values(data))[2] : data[2]
    blocks = (Paragraph(), Label(unique(eachobs(labels))))

    (recipe.split ? length(data) > 0 : numobs(data) > 0) ||
        error("No text files found in $textpath")
    return data, blocks
end


# Registering recipes

const RECIPES = Dict{String,Vector}(
    "imdb" =>
        [
            TextFolders(filefilterfn = f -> !occursin(r"tmp_clas|tmp_lm|unsup|test", f)),
            TextGenerationFolders(filefilterfn = f -> !occursin(r"tmp_clas|tmp_lm|test|rain", f))
        ]
)

## Tests

@testset "TextFolders [Recipe]" begin
    @test length(datarecipes(id = "imdb")) >= 1
end
