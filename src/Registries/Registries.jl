module Registries

using ..FastAI
using ..FastAI.Datasets: DatasetLoader, DataDepLoader, isavailable, loaddata

import Markdown
using DataDeps
using FeatureRegistries
using FeatureRegistries: Field

filterblocks(query) = supported -> FastAI.blocktypesmatch(supported, query)

include("datasets.jl")
include("tasks.jl")
include("recipes.jl")



export DATASETS,
    TASKS,
    DATARECIPES,
    find,
    info,
    load


function __init__()
    _registerdatasets(DATASETS)
end

end
