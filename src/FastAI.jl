module FastAI


using Reexport

@reexport using DLPipelines
@reexport using FluxTraining
@reexport using DataLoaders
@reexport using Flux

using Colors
using DataAugmentation
using DataAugmentation: getbounds, makebounds
using DLDatasets
using DLPipelines: methoddataset, methodmodel, methodlossfn
using LearnBase: getobs, nobs
using FixedPointNumbers
using Flux
using FluxTraining: Learner
using FluxModels
using MLDataPattern
using Parameters
using StaticArrays

const Datasets = DLDatasets
const Models = FluxModels

# method implementations and helpers
include("./steps/utils.jl")
include("./steps/step.jl")
include("./steps/spatial.jl")
include("./steps/imagepreprocessing.jl")
include("./methods/imageclassification.jl")

# training utilities
include("./datautils.jl")
include("training.jl")

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
    finetune!


end  # module
