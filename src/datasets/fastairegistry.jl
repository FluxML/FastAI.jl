
const FASTAI_DATA_RECIPES = Dict{String,Vector{DatasetRecipe}}(
    # Vision datasets
    [name => [ImageFolders()] for name in (
        "imagenette", "imagenette-160", "imagenette-320",
        "imagenette2", "imagenette2-160", "imagenette2-320",
        "imagewoof", "imagewoof-160", "imagewoof-320",
        "imagewoof2", "imagewoof2-160", "imagewoof2-320",
        "cifar10", "cifar100", "caltech_101", "mnist_png",
        "mnist_sample", "CUB_200_2011"
    )]...,
    [name => [ImageFolders(filefilterfn=f -> !(occursin("unsup", f)))]
        for name in ("imagewang-160", "imagewang-320", "imagewang")]...,
    "camvid" => [ImageSegmentationFolders()],
    "camvid_tiny" => [ImageSegmentationFolders()],
    "pascal_2007" => [ImageTableMultiLabel()],
    "mnist_tiny" => [ImageFolders(filefilterfn=f -> !occursin("test", f))],
    "mnist_var_size_tiny" => [ImageFolders(filefilterfn=f -> !occursin("test", f))],


    # Tabular datasets
    "adult_sample" => [
        TableDatasetRecipe(file="adult.csv"),
        TableClassificationRecipe(TableDatasetRecipe(file="adult.csv"), :salary),
        TableRegressionRecipe(TableDatasetRecipe(file="adult.csv"), :age),
    ],
)


"""
    const FASTAI_DATA_REGISTRY

The default `DataRegistry` containing every dataset in
the fastai dataset collection.
"""
const FASTAI_DATA_REGISTRY = DatasetRegistry(
    Dict(d => () -> datasetpath(d) for d in DATASETS),
    FASTAI_DATA_RECIPES,
)
