module Registries

using ..FastAI
using ..FastAI.Datasets: DatasetLoader, DataDepLoader, isavailable, loaddata

import Markdown
using DataDeps
using FeatureRegistries
using FeatureRegistries: Field

filterblocks(query) = supported -> FastAI.blocktypesmatch(supported, query)

_formatblock(t::Type{<:Tuple}) = _formatblock(Tuple(t.types))
_formatblock(t::Tuple) = map(_formatblock, t)
_formatblock(T::Type) = T
_formatblock(B::T) where T = T

include("datasets.jl")
include("tasks.jl")
include("recipes.jl")



export datasets,
    learningtasks,
    datarecipes,
    find,
    info,
    load


function __init__()
    _registerdatasets(DATASETS)
end

end
