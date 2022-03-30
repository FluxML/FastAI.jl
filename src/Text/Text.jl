module Text


using ..FastAI
using ..FastAI:
    # blocks
    Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
    LabelMulti, wrapped, Continuous, getencodings, getblocks, encodetarget, encodeinput,
    # encodings
    Encoding, StatefulEncoding, OneHot,
    # visualization
    ShowText,
    # other
    Context, Training, Validation, FASTAI_METHOD_REGISTRY, registerlearningtask!

import Requires: @require

include("recipes.jl")
include("blocks/text.jl")
include("transform.jl")

function __init__()
    _registerrecipes()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        import .Makie
        import .Makie: @recipe, @lift
        import .FastAI: ShowMakie
        include("makie.jl")
    end
end

export TextBlock, TextFolders, replace_all_caps, replace_sentence_case
end

