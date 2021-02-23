module Models

using ..FastAI

using BSON
using Flux
using Zygote
using DataDeps
using Metalhead

include("./datadeps.jl")

function __init__()
    initdatadeps()
end

include("./metalhead.jl")
include("./xresnet.jl")


export xresnet18, xresnet50


end
