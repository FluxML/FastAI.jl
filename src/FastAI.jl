module FastAI


using Reexport
@reexport using DLPipelines
@reexport using FluxTraining
@reexport using DataLoaders
@reexport using Flux

using Animations
using Makie
using Colors
using DataAugmentation
using DataAugmentation: getbounds, Bounds
import DLPipelines: methoddataset, methodmodel, methodlossfn, methoddataloaders,
    mockmodel, mocksample, predict, predictbatch, mockmodel, encode, encodeinput,
    encodetarget, decodeŷ, decodey
using LearnBase: getobs, nobs
using FilePathsBase
using FixedPointNumbers
using Flux
using Flux.Optimise
import Flux.Optimise: apply!, Optimiser, WeightDecay
using FluxTraining: Learner, handle
using FluxTraining.Events
using JLD2: jldsave, jldopen
using MLDataPattern
using Parameters
using StaticArrays
using ShowCases
using Test: @testset, @test, @test_nowarn

include("tasks.jl")
include("plotting.jl")
include("learner.jl")

# method implementations and helpers
include("datablock/block.jl")
include("datablock/encoding.jl")
include("datablock/method.jl")
include("datablock/models.jl")
include("datablock/loss.jl")

include("encodings/onehot.jl")
include("encodings/imagepreprocessing.jl")
include("encodings/projective.jl")
include("encodings/scalepoints.jl")

#=
include("./methods/imageclassification.jl")
include("./methods/imagesegmentation.jl")
include("./methods/singlekeypointregression.jl")
include("./methods/checks.jl")
=#


# submodules
include("datasets/Datasets.jl")
@reexport using .Datasets


include("models/Models.jl")
using .Models

# training
include("training/paramgroups.jl")
include("training/discriminativelrs.jl")
include("training/utils.jl")
include("training/onecycle.jl")
include("training/finetune.jl")
include("training/lrfind.jl")

include("serialization.jl")




export
    # submodules
    Datasets,
    Models,
    datasetpath,
    loadtaskdata,
    mapobs,
    groupobs,
    filterobs,
    shuffleobs,
    datasubset,

    # method API
    methodmodel,
    methoddataset,
    methoddataloaders,
    methodlossfn,
    getobs,
    nobs,
    predict,
    predictbatch,

    # plotting API
    plotbatch,
    plotsamples,
    plotpredictions,
    makebatch,

    # encodings
    ProjectiveTransforms,
    ImagePreprocessing,
    OneHot,
    ScalePoints,
    augs_projection, augs_lighting,

    BlockMethod,

    # training
    methodlearner,
    Learner,
    fit!,
    fitonecycle!,
    finetune!,
    lrfind,
    savemethodmodel,
    loadmethodmodel,

    gpu,
    plot






end  # module
