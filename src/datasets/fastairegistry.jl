
"""
    const FASTAI_DATA_REGISTRY

The default `DataRegistry` containing every dataset in
the fastai dataset collection.
"""
const FASTAI_DATA_REGISTRY = DatasetRegistry(
    Dict(d => () -> datasetpath(d) for d in DATASETS),
    Dict{String,Vector{DatasetRecipe}}(),
)
