module FastAI

using DLDatasets
using DLPipelines
using DLPipelines: methodmodel, methoddataset, methoddataloaders, ImageClassification
using DataAugmentation
using DataLoaders
using Flux
using FluxTraining
using MLDataPattern

const Datasets = DLDatasets


include("training.jl")


export
    # submodules
    Datasets,
    loaddataset,

    # method API
    methodmodel,
    methoddataset,
    methoddataloaders,

    # methods
    ImageClassification,

    # training
    fit!,
    finetune!


end  # module
