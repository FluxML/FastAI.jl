"""
    module Datasets

Commonly used datasets and utilities for creating data containers.

In the future, contents will be integrated into packages:

- FastAI datasets and data containers will be moved into MLDatasets.jl
- data container transformations will be moved to MLDataPattern.jl

This submodule will then reexport the same definitions.
"""
module Datasets


using ..FastAI
using ..FastAI: typify

using DataDeps
using Glob
using FilePathsBase
import DataAugmentation
using FilePathsBase: filename
import FileIO
using IndirectArrays: IndirectArray
using MLDataPattern
using MLDataPattern: splitobs
import LearnBase
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
include("transformations.jl")

include("load.jl")
include("recipe.jl")

include("deprecations.jl")

include("loaders.jl")


export
    # reexports from MLDataPattern
    splitobs,

    # container transformations
    mapobs,
    filterobs,
    groupobs,
    joinobs,
    eachobs,

    # primitive containers
    FileDataset,
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
    #DATASETS,
    loadfolderdata,
    datasetpath,

    # recipes
    loadrecipe,
    finddatasets,
    listdatasources,
    loaddataset

end  # module
