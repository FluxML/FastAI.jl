module Datasets

using ..FastAI
using ..FastAI: typify

import MLUtils: MLUtils, getobs, numobs, filterobs, groupobs, mapobs,
    shuffleobs, groupobs, eachobs, ObsView
import MLDatasets
using MLUtils: mapobs, groupobs
using DataDeps
using FilePathsBase
import DataAugmentation
using FilePathsBase: filename
import FileIO
using IndirectArrays: IndirectArray
using Colors
using FixedPointNumbers
using DataFrames
using Tables
using CSV
using ShowCases
using InlineTest

include("fastaidatasets.jl")

include("batching.jl")
include("containers.jl")
include("load.jl")
include("loaders.jl")
include("recipe.jl")

include("deprecations.jl")

function __init__()
    initdatadeps()
end

export
    # primitive containers
    TableDataset,
    TimeSeriesDataset,

    mapobs, eachobs, groupobs, shuffleobs, ObsView,

    # utilities
    isimagefile,
    istextfile,
    istimeseriesfile,
    matches,
    loadfile,
    loadmask,
    pathname,
    pathparent,
    parentname,
    grandparentname,

    # datasets
    loadfolderdata,
    datasetpath,

    # recipes
    loadrecipe,
    finddatasets,
    listdatasources,
    loaddataset

end  # module
