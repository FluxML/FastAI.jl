
const FASTAI_DATA_RECIPES = Dict{String, Vector{DatasetRecipe}}(
    # Image classification datasets
    [name => [ImageClassificationFolders()] for name in (
        "imagenette", "imagenette-160", "imagenette-320",
        "imagenette2", "imagenette2-160", "imagenette2-320",
        "imagewoof", "imagewoof-160", "imagewoof-320",
        "imagewoof2", "imagewoof2-160", "imagewoof2-320",
    )]...,

    "camvid_tiny" => [ImageSegmentationFolders()],
)


"""
    const FASTAI_DATA_REGISTRY

The default `DataRegistry` containing every dataset in
the fastai dataset collection.
"""
const FASTAI_DATA_REGISTRY = DatasetRegistry(
    Dict(d => () -> datasetpath(d) for d in FastAI.DATASETS),
    FASTAI_DATA_RECIPES,
)
