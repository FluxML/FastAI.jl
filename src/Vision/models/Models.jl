module Models


using Base: Bool, Symbol
using ..FastAI

using BSON
using Flux
using Zygote
using DataDeps
using InlineTest


include("layers.jl")
include("blocks.jl")
include("xresnet.jl")
include("unet.jl")


export xresnet18, xresnet50, UNetDynamic, TabularModel

end
