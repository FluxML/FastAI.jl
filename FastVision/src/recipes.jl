
# ## [`DatasetRecipe`] implementations

"""
    ImageFolders(; labelfn = parentname, split = false)

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
Base.@kwdef struct ImageFolders <: Datasets.DatasetRecipe
    labelfn = Datasets.parentname
    split::Bool = false
    filefilterfn = _ -> true
end

function Datasets.loadrecipe(recipe::ImageFolders, path)
    isdir(path) || error("$path is not a directory")
    data = loadfolderdata(path,
                          filterfn = f -> isimagefile(f) && recipe.filefilterfn(f),
                          loadfn = (loadfile, recipe.labelfn),
                          splitfn = recipe.split ? grandparentname : nothing)

    (recipe.split ? length(data) > 0 : numobs(data) > 0) ||
        error("No image files found in $path")

    labels = recipe.split ? first(values(data))[2] : data[2]
    blocks = Image{2}(), Label(unique(eachobs(labels)))
    length(blocks[2].classes) > 1 ||
        error("Expected multiple different labels, got: $(blocks[2].classes))")
    return data, blocks
end

Datasets.recipeblocks(::Type{ImageFolders}) = Tuple{Image{2}, Label}

"""
    ImageSegmentationFolders(; imagefolder="images", maskfolder="labels", labelfile="codes.txt")

Dataset recipe for loading 2D image segmentation datasets from a common format
where images and masks are stored as images in two different subfolders
`<root>/<imagefolder>` and `<root>/<maskfolder>`
The class labels should be in a newline-delimited file `<root>/<labelfile>`.
"""
Base.@kwdef struct ImageSegmentationFolders <: Datasets.DatasetRecipe
    imagefolder::String = "images"
    maskfolder::String = "labels"
    labelfile::String = "codes.txt"
end

function Datasets.loadrecipe(recipe::ImageSegmentationFolders, path)
    isdir(path) || error("$path is not a directory")
    imagepath = joinpath(path, recipe.imagefolder)
    maskpath = joinpath(path, recipe.maskfolder)
    classespath = joinpath(path, recipe.labelfile)

    isdir(imagepath) || error("Image folder $imagepath is not a directory")
    isdir(maskpath) || error("Mask folder $maskpath is not a directory")

    isfile(classespath) || error("Classes file $classespath does not exist")
    classes = readlines(open(joinpath(path, recipe.labelfile)))
    length(classes) > 1 ||
        error("Expected multiple different labels, got: $(blocks[2].classes))")

    images = loadfolderdata(imagepath, filterfn = isimagefile, loadfn = loadfile)
    masks = loadfolderdata(maskpath, filterfn = isimagefile,
                           loadfn = f -> loadmask(f, classes))
    numobs(images) == numobs(masks) ||
        error("Expected the same number of images and masks, but found $(numobs(images)) images and $(numobs(masks)) masks")
    numobs(images) > 0 ||
        error("No images or masks found in folders $imagepath and $maskpath")

    blocks = Image{2}(), Mask{2}(classes)
    return (images, masks), blocks
end

Datasets.recipeblocks(::Type{ImageSegmentationFolders}) = Tuple{Image{2}, Mask{2}}

# ImageTableMultiLabel

Base.@kwdef struct ImageTableMultiLabel <: Datasets.DatasetRecipe
    csvfile::String = "train.csv"
    imagefolder::String = "train"
    filecol::Symbol = :fname
    fileext::String = ""
    labelcol::Symbol = :labels
    split::Bool = false
    splitcol::Symbol = :is_valid
    labelsep::String = " "
end

function Datasets.loadrecipe(recipe::ImageTableMultiLabel, path)
    csvpath = joinpath(path, recipe.csvfile)
    isfile(csvpath) || error("File $csvpath does not exist")
    df = loadfile(csvpath)
    images = mapobs(f -> loadfile(joinpath(path, recipe.imagefolder, f * recipe.fileext)),
                    df[:, recipe.filecol])
    labels = map(str -> split(str, recipe.labelsep),
                 df[:, recipe.labelcol])
    data = (images, labels)
    blocks = Image{2}(), LabelMulti(unique(Iterators.flatten(labels)))
    if recipe.split
        idxs = 1:numobs(data)
        splits = df[:, recipe.splitcol]
        data = Dict("train" => MLUtils.ObsView(data, idxs[splits]),
                    "valid" => MLUtils.ObsView(data, idxs[(!).(splits)]))
    end
    return data, blocks
end

Datasets.recipeblocks(::Type{ImageTableMultiLabel}) = Tuple{Image{2}, LabelMulti}

# ## Registering recipes for fastai datasets

const RECIPES = Dict{String, Vector}([name => [ImageFolders()]
                                      for name in ("imagenette",
                                                   "imagenette-160",
                                                   "imagenette-320",
                                                   "imagenette2",
                                                   "imagenette2-160",
                                                   "imagenette2-320",
                                                   "imagewoof",
                                                   "imagewoof-160",
                                                   "imagewoof-320",
                                                   "imagewoof2",
                                                   "imagewoof2-160",
                                                   "imagewoof2-320",
                                                   "cifar10",
                                                   "cifar100",
                                                   "caltech_101",
                                                   "mnist_png",
                                                   "mnist_sample",
                                                   "CUB_200_2011",
                                                   "food-101")]...,
                                     [name => [
                                          ImageFolders(filefilterfn = f -> !(occursin("unsup",
                                                                                      f))),
                                      ]
                                      for name in ("imagewang-160",
                                                   "imagewang-320",
                                                   "imagewang")]...,
                                     "camvid" => [
                                         ImageSegmentationFolders(),
                                     ],
                                     "camvid_tiny" => [
                                         ImageSegmentationFolders(),
                                     ],
                                     "pascal_2007" => [
                                         ImageTableMultiLabel(),
                                     ],
                                     "mnist_tiny" => [
                                         ImageFolders(filefilterfn = f -> !occursin("test",
                                                                                    f)),
                                     ],
                                     "mnist_var_size_tiny" => [
                                         ImageFolders(filefilterfn = f -> !occursin("test",
                                                                                    f)),
                                     ])

# ## Tests

@testset "ImageFolders [recipe]" begin
    path = joinpath(load(datasets()["mnist_var_size_tiny"]), "train")

    @testset "Basic configuration" begin
        recipe = ImageFolders()
        data, blocks = Datasets.loadrecipe(recipe, path)
        Datasets.testrecipe(recipe, data, blocks)
        @test blocks[1] isa Image
        @test blocks[2].classes == ["3", "7"]
    end

    @testset "Split configuration" begin
        recipe = ImageFolders(split = true)
        data, blocks = Datasets.loadrecipe(recipe, path)
        Datasets.testrecipe(recipe, data["train"], blocks)
    end

    @testset "Error cases" begin
        @testset "Empty directory" begin
            recipe = ImageFolders(split = true)
            @test_throws ErrorException Datasets.loadrecipe(recipe, mktempdir())
        end

        @testset "Only one label" begin
            recipe = ImageFolders(labelfn = x -> "1")
            @test_throws ErrorException Datasets.loadrecipe(recipe, path)
        end
    end
end

@testset "ImageSegmentationFolders [recipe]" begin
    path = load(datasets()["camvid_tiny"])

    @testset "Basic configuration" begin
        recipe = ImageSegmentationFolders()
        data, blocks = Datasets.loadrecipe(recipe, path)
        Datasets.testrecipe(recipe, data, blocks)
        @test blocks[1] isa Image
        @test blocks[2] isa Mask
    end

    @testset "Error cases" begin
        @testset "Empty directory" begin
            recipe = ImageSegmentationFolders()
            @test_throws ErrorException Datasets.loadrecipe(recipe, mktempdir())
        end

        @testset "Only one label" begin
            recipe = ImageSegmentationFolders(labelfile = "idontexist")
            @test_throws ErrorException Datasets.loadrecipe(recipe, path)
        end
    end
end
