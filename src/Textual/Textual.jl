module Textual


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
    Context, Training, Validation

import Requires: @require

using InlineTest
using Random

include("recipes.jl")
include("blocks/text.jl")
include("transform.jl")
include("tasks/classification.jl")


function __init__()
    _registerrecipes()
end

export Paragraph,
# learning tasks
TextClassficationSingle
# encodings
replace_all_caps, replace_sentence_case, convert_lowercase
end
