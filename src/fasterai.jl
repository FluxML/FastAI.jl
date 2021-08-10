defaultdataregistry() = Datasets.FASTAI_DATA_REGISTRY
defaultmethodregistry() = LearningMethodRegistry()


Datasets.listdatasources() = listdatasources(defaultdataregistry())
Datasets.finddatasets(; kwargs...) = finddatasets(defaultdataregistry(); kwargs...)
Datasets.loaddataset(args...; kwargs...) = loaddataset(defaultdataregistry(), args...; kwargs...)
