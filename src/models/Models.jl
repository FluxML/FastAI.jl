module Models

using ..FastAI

using BSON
using Flux
using Zygote
using DataDeps


include("layers.jl")
include("blocks.jl")

include("xresnet.jl")
include("unet.jl")


export xresnet18, xresnet50, UNetDynamic


end
