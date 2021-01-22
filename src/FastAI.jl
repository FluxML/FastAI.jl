module FastAI

using DLPipelines
using DataAugmentation
using DataLoaders
using Flux
using FluxTraining


include("training.jl")


export
    # method API
    methodmodel,
    methoddataset,
    methoddataloaders,

    # training
    fit!,
    finetune!


end  # module
