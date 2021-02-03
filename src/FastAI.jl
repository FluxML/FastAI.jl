module FastAI


using Reexport

@reexport using DLPipelines
@reexport using FluxTraining
@reexport using DataLoaders
@reexport using Flux

using Animations
using Colors
using DataAugmentation
using DataAugmentation: getbounds, makebounds
using DLPipelines: methoddataset, methodmodel, methodlossfn, methoddataloaders
using LearnBase: getobs, nobs
using FixedPointNumbers
using Flux
using FluxTraining: Learner, handle
using FluxTraining.Events
using FluxModels
using MLDataPattern
using Parameters
using StaticArrays

const Models = FluxModels

# method implementations and helpers
include("./steps/utils.jl")
include("./steps/step.jl")
include("./steps/spatial.jl")
include("./steps/imagepreprocessing.jl")
include("./methods/imageclassification.jl")

# submodules
include("datasets/Datasets.jl")
using .Datasets

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
    loaddataset,

    # method API
    methodmodel,
    methoddataset,
    methoddataloaders,
    methodlossfn,
    getobs,
    nobs,


    # pipeline steps
    ProjectiveTransforms, ImagePreprocessing,

    # methods
    ImageClassification,

    # training
    Learner,
    fit!,
    fitonecycle!,
    finetune!


end  # module
