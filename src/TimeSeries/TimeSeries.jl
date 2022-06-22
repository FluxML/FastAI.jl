module TimeSeries


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
import ..FastAI: Datasets
using ..FastAI.Datasets
# for tests
using ..FastAI: testencoding

# extending
import ..FastAI:
    blockmodel, blockbackbone, blocklossfn, encode, decode, checkblock,
    encodedblock, decodedblock, showblock!, mockblock, setup

import MLUtils: MLUtils, eachobs, getobs, numobs
import Requires: @require

using FilePathsBase
using InlineTest

# Blocks
include("blocks/timeseriesrow.jl")

include("recipes.jl")

function __init__()
    _registerrecipes()
end

export TimeSeriesRow

end