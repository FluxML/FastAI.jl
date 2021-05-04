
using Colors: RGB, N0f8
using FastAI
using FastAI: ParamGroups, IndexGrouper, getgroup, DiscriminativeLRs
using FilePathsBase
using FastAI.Datasets
using DLPipelines
using DataAugmentation
using DataAugmentation: getbounds
using Flux
using Flux.Optimise: Optimiser, apply!
using StaticArrays
using Test
using TestSetExtensions

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
include("testdata.jl")
