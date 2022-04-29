module TimeSeries

using ..FastAI
using ..FastAI:
    # blocks
    Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
    LabelMulti, wrapped, Continuous, getencodings, getblocks,
    # encodings
    Encoding, StatefulEncoding, OneHot,
    # visualization
    ShowText,
    # other
    FASTAI_METHOD_REGISTRY, registerlearningmethod!

# for tests
using ..FastAI: testencoding

# extending
import ..FastAI:
    blockmodel, blockbackbone, blocklossfn, encode, decode, checkblock,
    encodedblock, decodedblock, showblock!, mockblock, setup

import Requires: @require
import DataFrames: DataFrame, Not, select
import UnicodePlots
import ARFFFiles

using FilePathsBase
using InlineTest

# Blocks
include("blocks/timeseriesrow.jl")

include("recipes.jl")

export TimeSeriesRow

end