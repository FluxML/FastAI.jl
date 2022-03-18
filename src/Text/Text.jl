module Text

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

import DataAugmentation
import DataFrames: DataFrame
import Flux: Embedding, Chain, Dropout, Dense, Parallel
import PrettyTables
import Requires: @require
import ShowCases: ShowCase
import Tables
import Statistics

using InlineTest


# Blocks
include("blocks/textrow.jl")

export TextRow

end
