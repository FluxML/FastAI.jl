module FastAI


using Base: NamedTuple
using Colors: colormaps_sequential
using Reexport
@reexport using DLPipelines
@reexport using FluxTraining
@reexport using DataLoaders
@reexport using Flux

using Animations
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
import ImageInTerminal
using MLDataPattern
using Parameters
using PrettyTables
using Requires
using StaticArrays
using Setfield
using ShowCases
using Tables
import Test
import UnicodePlots
using Statistics
using InlineTest

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

# Blocks
include("blocks/label.jl")

include("blocks/bounded.jl")

# Encodings
include("encodings/tabularpreprocessing.jl")
include("encodings/onehot.jl")
include("encodings/imagepreprocessing.jl")
include("encodings/projective.jl")
include("encodings/keypointpreprocessing.jl")

# Training interface
include("datablock/models.jl")
include("datablock/loss.jl")

# Interpretation
include("interpretation/backend.jl")
include("interpretation/text.jl")
include("interpretation/detect.jl")
include("interpretation/method.jl")
include("interpretation/showinterpretable.jl")
include("interpretation/learner.jl")

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


function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        using .Makie
        include("interpretation/makie/recipes.jl")
        include("interpretation/makie/showmakie.jl")
        include("interpretation/makie/lrfind.jl")
    end
end



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
    setup,
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
    makebatch,
    getbatch,

    # interpretation
    ShowText,
    ShowMakie,
    showblock,
    showblocks,
    showsample,
    showsamples,
    showoutput,
    showoutputs,
    showoutputbatch,
    showencodedsample,
    showencodedsamples,
    showbatch,
    showprediction,
    showpredictions,

    # learning methods
    findlearningmethods,
    ImageClassificationSingle,
    ImageClassificationMulti,
    ImageSegmentation,
    ImageKeypointRegression,
    TabularClassificationSingle,
    TabularRegression,


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
