module Datasets

using ..FastAI
using ..FastAI: typify

import MLUtils: MLUtils, getobs, numobs, filterobs, groupobs, mapobs,
                shuffleobs, groupobs, eachobs, ObsView
import MLDatasets
using MLUtils: mapobs, groupobs
using DataDeps
using FilePathsBase
using FilePathsBase: filename
import FileIO
using InlineTest

include("fastaidatasets.jl")

include("batching.jl")
include("load.jl")
include("loaders.jl")
include("recipe.jl")

function __init__()
    initdatadeps()
end

export
# primitive containers
      mapobs, eachobs, groupobs, shuffleobs, ObsView,

# utilities
      matches,
      loadfile,
      pathname,
      pathparent,
      parentname,
      grandparentname,

# datasets
      loadfolderdata,
      datasetpath

end  # module
