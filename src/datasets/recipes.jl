"""
abstract type DatasetRecipe

A recipe that contains configuration for loading a data container. Calling it with a path returns a data container and the blocks that each sample is made of.

#### Interface

- `loadrecipe(::DatasetRecipe, args...) -> (data, blocks)`
- `recipeblocks(::Type{DatasetRecipe}) -> TBlocks`

#### Invariants

- `data` must be a data container of samples that are valid `blocks`, i.e. `checkblock(blocks, getobs(data, 1)) == true`
"""
abstract type DatasetRecipe end


"""
    loadrecipe(recipe, path)

Load a recipe from a path. Return a data container `data` and concrete
`blocks`.
"""
function loadrecipe end


"""
    recipeblocks(TRecipe) -> TBlocks
    recipeblocks(recipe) -> TBlocks

Return the `Block` _types_ for the data container that recipe
type `TRecipe` creates. Does not return `Block` instances as the exact
configuration may not be known until the dataset is being
loaded.

#### Examples

```julia
recipeblocks(ImageLabelClf) == Tuple{Image{2}, Label}
```
"""
recipeblocks(::R) where {R<:DatasetRecipe} = recipeblocks(R)


# ## Implementations

# ImageClfFolders

"""
    ImageClfFolders(; labelfn = parentname, split = false)

Recipe for loading a single-label image classification dataset
stored in a hierarchical folder format. If `split == true`, split
the data container on the name of the grandparent folder. The label
defaults to the name of the parent folder but a custom function can
be passed as `labelfn`.

```julia
julia> recipeblocks(ImageFolders)
Tuple{Image{2}, Label}
```
"""
Base.@kwdef struct ImageFolders <: DatasetRecipe
    labelfn = parentname
    split::Bool = false
end

function loadrecipe(recipe::ImageFolders, path)
    isdir(path) || error("$path is not a directory")
    data = loadfolderdata(
        path,
        filterfn=isimagefile,
        loadfn=(loadfile, recipe.labelfn),
        splitfn=recipe.split ? grandparentname : nothing)

    (recipe.split ? length(data) > 0 : nobs(data) > 0) || error("No image files found in $path")

    labels = recipe.split ? first(values(data))[2] : data[2]
    blocks = Image{2}(), Label(unique(eachobs(labels)))
    length(blocks[2].classes) > 1 || error("Expected multiple different labels, got: $(blocks[2].classes))")
    return data, blocks
end

recipeblocks(::Type{ImageFolders}) = Tuple{Image{2}, Label}


# ImageSegmentationFolders


"""
    ImageSegmentationFolders(; imagefolder="images", maskfolder="labels", labelfile="codes.txt")

Dataset recipe for loading 2D image segmentation datasets from a common format
where images and masks are stored as images in two different subfolders
"<root>/<imagefolder>" and "<root>/<maskfolder>"
The class labels should be in a newline-delimited file "<root>/<labelfile>".
"""
Base.@kwdef struct ImageSegmentationFolders <: DatasetRecipe
    imagefolder::String = "images"
    maskfolder::String = "labels"
    labelfile::String = "codes.txt"
end

function loadrecipe(recipe::ImageSegmentationFolders, path)
    isdir(path) || error("$path is not a directory")
    imagepath = joinpath(path, recipe.imagefolder)
    maskpath = joinpath(path, recipe.maskfolder)
    classespath = joinpath(path, recipe.labelfile)

    isdir(imagepath) || error("Image folder $imagepath is not a directory")
    isdir(maskpath) || error("Mask folder $maskpath is not a directory")

    isfile(classespath) || error("Classes file $classespath does not exist")
    classes = readlines(open(joinpath(path, recipe.labelfile)))
    length(classes) > 1 || error("Expected multiple different labels, got: $(blocks[2].classes))")

    images = loadfolderdata(imagepath, filterfn=isimagefile, loadfn=loadfile)
    masks = loadfolderdata(maskpath, filterfn=isimagefile, loadfn=f -> loadmask(f, classes))
    nobs(images) == nobs(masks) || error("Expected the same number of images and masks, but found $(nobs(images)) images and $(nobs(masks)) masks")
    nobs(images) > 0 || error("No images or masks found in folders $imagepath and $maskpath")

    blocks = Image{2}(), Mask{2}(classes)
    return (images, masks), blocks
end

recipeblocks(::Type{ImageSegmentationFolders}) = Tuple{Image{2}, Mask{2}}

# ImageTableMultiLabel

Base.@kwdef struct ImageTableMultiLabel <: DatasetRecipe
    csvfile::String = "train.csv"
    imagefolder::String = "train"
    filecol::Symbol = :fname
    labelcol::Symbol = :labels
    split::Bool = false
    splitcol::Symbol = :is_valid
    labelsep::String = " "
end


function loadrecipe(recipe::ImageTableMultiLabel, path)
    csvpath = joinpath(path, recipe.csvfile)
    isfile(csvpath) || error("File $csvpath does not exist")
    df = loadfile(csvpath)
    images = mapobs(f -> loadfile(joinpath(path, recipe.imagefolder, f)), df[:, recipe.filecol])
    labels = map(str -> split(str, recipe.labelsep), df[:,recipe.labelcol])
    data = (images, labels)
    blocks = Image{2}(), LabelMulti(unique(Iterators.flatten(labels)))
    if recipe.split
        idxs = 1:nobs(data)
        splits = df[:, recipe.splitcol]
        data = Dict(
            "train" => datasubset(data, idxs[splits]),
            "valid" => datasubset(data, idxs[(!).(splits)])
        )
    end
    return data, blocks
end

recipeblocks(::Type{ImageTableMultiLabel}) = Tuple{Image{2}, LabelMulti}
