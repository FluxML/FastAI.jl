
using Colors: RGB, N0f8
using FastAI
using FilePathsBase
using FastAI.Datasets
using DLPipelines
using DataAugmentation
using DataAugmentation: getbounds
using Flux
using StaticArrays
using Test
using TestSetExtensions

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
include("testdata.jl")
