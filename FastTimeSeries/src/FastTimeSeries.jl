module FastTimeSeries


using FastAI
using FastAI:
    # blocks
    Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
    LabelMulti, wrapped, Continuous, getencodings, getblocks, encodetarget, encodeinput,
    # encodings
    Encoding, StatefulEncoding, OneHot,
    # visualization
    ShowText,
    # other
    Context, Training, Validation
using FastAI.Datasets
# for tests
using ..FastAI: testencoding

# extending
import FastAI:
    blockmodel, blockbackbone, blocklossfn, encode, decode, checkblock,
    encodedblock, decodedblock, showblock!, mockblock, setup

import MLUtils: MLUtils, eachobs, getobs, numobs

using FilePathsBase
using InlineTest
using Statistics
using UnicodePlots
using Flux

# Blocks
include("blocks/timeseriesrow.jl")

# Encodings
include("encodings/tspreprocessing.jl")
include("encodings/continuouspreprocessing.jl")

# Models
include("models/Models.jl")
include("models.jl")

include("container.jl")
include("recipes.jl")

const _tasks = Dict{String, Any}()
include("tasks/classification.jl")
include("tasks/regression.jl")

function __init__()
    _registerrecipes()
    foreach(values(_tasks)) do t
        if !haskey(FastAI.learningtasks(), t.id)
            push!(FastAI.learningtasks(), t)
        end
    end
end

export
    TimeSeriesRow, TSClassificationSingle, TSPreprocessing, TSRegression
end
