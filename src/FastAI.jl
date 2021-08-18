module FastAI


using Base: NamedTuple
using Colors: colormaps_sequential
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
using IndirectArrays: IndirectArray
using LearnBase: getobs, nobs
using FilePathsBase
using FixedPointNumbers
using Flux
using Flux.Optimise
import Flux.Optimise: apply!, Optimiser, WeightDecay
using FluxTraining: Learner, handle
using FluxTraining.Events
using JLD2: jldsave, jldopen
using Markdown
using MLDataPattern
using Parameters
using PrettyTables
using StaticArrays
using Setfield
using ShowCases
using Tables
using Statistics
using Test: @testset, @test, @test_nowarn

include("plotting.jl")
include("learner.jl")

# Data block API
include("datablock/block.jl")
include("datablock/encoding.jl")
include("datablock/method.jl")
include("datablock/describe.jl")
include("datablock/checks.jl")
include("datablock/wrappers.jl")

# submodules
include("datasets/Datasets.jl")
@reexport using .Datasets

include("models/Models.jl")
using .Models

# Encodings
include("encodings/tabularpreprocessing.jl")
include("encodings/onehot.jl")
include("encodings/imagepreprocessing.jl")
include("encodings/projective.jl")
include("encodings/keypointpreprocessing.jl")

# Training interface
include("datablock/models.jl")
include("datablock/loss.jl")
include("datablock/plot.jl")

# training
include("training/paramgroups.jl")
include("training/discriminativelrs.jl")
include("training/utils.jl")
include("training/onecycle.jl")
include("training/finetune.jl")
include("training/lrfind.jl")
include("training/metrics.jl")

include("serialization.jl")


include("fasterai/methodregistry.jl")
include("fasterai/learningmethods.jl")
include("fasterai/defaults.jl")




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
    plotsample,
    plotsamples,
    plotpredictions,
    makebatch,

    # blocks
    Image,
    Mask,
    Label,
    LabelMulti,
    Keypoints,
    Many,
    TableRow,
    Continuous,

    # encodings
    encode,
    decode,
    ProjectiveTransforms,
    ImagePreprocessing,
    OneHot,
    KeypointPreprocessing,
    Only,
    Named,
    augs_projection, augs_lighting,
    TabularPreprocessing,

    BlockMethod,
    describemethod,
    checkblock,
    
    # learning methods
    findlearningmethods,
    ImageClassificationSingle,
    ImageClassificationMulti,
    ImageSegmentation,
    ImageKeypointRegression,


    # training
    methodlearner,
    Learner,
    fit!,
    fitonecycle!,
    finetune!,
    lrfind,
    savemethodmodel,
    loadmethodmodel,
    accuracy_thresh,

    gpu,
    plot






end  # module
