defaultdataregistry() = Datasets.FASTAI_DATA_REGISTRY
defaulttaskregistry() = FASTAI_METHOD_REGISTRY


Datasets.listdatasources() = listdatasources(defaultdataregistry())
Datasets.finddatasets(; kwargs...) = finddatasets(defaultdataregistry(); kwargs...)
Datasets.loaddataset(args...; kwargs...) = loaddataset(defaultdataregistry(), args...; kwargs...)

findlearningtasks(args...; kwargs...) = findlearningtasks(defaulttaskregistry(), args...; kwargs...)
