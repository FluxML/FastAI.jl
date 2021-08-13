defaultdataregistry() = Datasets.FASTAI_DATA_REGISTRY
defaultmethodregistry() = FASTAI_METHOD_REGISTRY


Datasets.listdatasources() = listdatasources(defaultdataregistry())
Datasets.finddatasets(; kwargs...) = finddatasets(defaultdataregistry(); kwargs...)
Datasets.loaddataset(args...; kwargs...) = loaddataset(defaultdataregistry(), args...; kwargs...)

findlearningmethods(args...; kwargs...) = findlearningmethods(defaultmethodregistry(), args...; kwargs...)
