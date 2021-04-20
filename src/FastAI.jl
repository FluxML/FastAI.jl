module FastAI


using Reexport

@reexport using DLPipelines
@reexport using FluxTraining
@reexport using DataLoaders
@reexport using Flux

using Animations
using AbstractPlotting
using Colors
using DataAugmentation
using DataAugmentation: getbounds, makebounds
using DLPipelines: methoddataset, methodmodel, methodlossfn, methoddataloaders
using LearnBase: getobs, nobs
using FilePathsBase
using FixedPointNumbers
using Flux
using FluxTraining: Learner, handle
using FluxTraining.Events
using MLDataPattern
using Parameters
using StaticArrays

include("tasks.jl")
include("plotting.jl")
include("learner.jl")

# method implementations and helpers
include("./steps/utils.jl")
include("./steps/step.jl")
include("./steps/spatial.jl")
include("augmentation.jl")
include("./steps/imagepreprocessing.jl")
include("./methods/imageclassification.jl")
include("./methods/imagesegmentation.jl")


# submodules
include("datasets/Datasets.jl")
using .Datasets

include("models/Models.jl")
using .Models

# training
include("training/utils.jl")
include("training/onecycle.jl")
include("training/finetune.jl")
include("training/lrfind.jl")

export methodlossfn


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

    # method API
    methodmodel,
    methoddataset,
    methoddataloaders,
    methodlossfn,
    getobs,
    nobs,

    # plotting API
    plotbatch,
    plotsamples,

    # pipeline steps
    ProjectiveTransforms, ImagePreprocessing, augs_projection,

    # tasks
    ImageClassificationTask,
    ImageSegmentationTask,

    # methods
    ImageClassification,
    ImageSegmentation,

    # training
    methodlearner,
    Learner,
    fit!,
    fitonecycle!,
    finetune!,

    gpu






end  # module
