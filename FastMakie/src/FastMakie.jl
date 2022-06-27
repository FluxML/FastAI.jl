module FastMakie

import FastAI: FastAI, createhandle, showblock!, AbstractBlock, OneHotTensor,
               OneHotTensorMulti,
               Label, LabelMulti, ShowBackend, ShowMakie, axiskwargs, showblock,
               showblocks!, showblocks
using InlineTest
import Makie
import NNlib: softmax, sigmoid

include("axis.jl")
include("showmakie.jl")
include("blocks.jl")
include("lrfind.jl")

function __init__()
    # When loaded, set default show backend
    FastAI.SHOW_BACKEND[] = ShowMakie()
end

end
