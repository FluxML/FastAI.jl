module Models

using Base: Bool, Symbol
using ..FastAI

using BSON
using Flux
using Zygote
using DataDeps


include("layers.jl")
include("blocks.jl")

include("xresnet.jl")
include("unet.jl")
include("tabularmodel.jl")


export xresnet18, xresnet50, UNetDynamic, 
TabularModel, get_emb_sz, embeddingbackbone, continuousbackbone, classifierbackbone, sigmoidrange


end
