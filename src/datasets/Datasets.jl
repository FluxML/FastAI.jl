module Datasets

using ..FastAI
using ..FastAI: typify

import MLUtils: MLUtils, getobs, numobs, filterobs, groupobs, mapobs
import MLDatasets: FileDataset
using MLUtils: mapobs, groupobs
using DataDeps
using Glob
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

function __init__()
    initdatadeps()
end

include("containers.jl")
include("batching.jl")

include("load.jl")
include("recipe.jl")

include("deprecations.jl")

include("loaders.jl")


export
    # primitive containers
    TableDataset,

    # utilities
    isimagefile,
    istextfile,
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
