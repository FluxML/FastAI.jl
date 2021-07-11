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

using DataDeps
using Glob
using FilePathsBase
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

include("fastaidatasets.jl")

function __init__()
    initdatadeps()
end

include("containers.jl")
include("transformations.jl")

include("load.jl")


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
    matches,
    loadfile,
    loadmask,
    pathname,
    pathparent,
    parentname,
    grandparentname,

    # datasets
    DATASETS,
    loadtaskdata,
    datasetpath

end  # module
