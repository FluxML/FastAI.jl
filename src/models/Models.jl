module Models

using ..FastAI

using BSON
using Flux
using Zygote
using DataDeps
using Metalhead


function __init__()
    initdatadeps()
end

include("layers.jl")
include("blocks.jl")

include("./xresnet.jl")


export xresnet18, xresnet50


end
