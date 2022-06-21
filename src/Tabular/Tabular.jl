module Tabular


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
    Datasets,
    blockmodel, blockbackbone, blocklossfn, encode, decode, checkblock,
    encodedblock, decodedblock, showblock!, mockblock, setup
using ..FastAI.Datasets


import DataAugmentation
import DataFrames: DataFrame
import MLUtils: MLUtils, eachobs, getobs, numobs
import Flux
import Flux: Embedding, Chain, Dropout, Dense, Parallel, BatchNorm
import PrettyTables
import Requires: @require
import ShowCases: ShowCase
import Tables
import Statistics

using InlineTest


# Blocks
include("blocks/tablerow.jl")

# Encodings
include("encodings/tabularpreprocessing.jl")


include("models.jl")

const _tasks = Dict{String, Any}()
include("tasks/classification.jl")
include("tasks/regression.jl")
include("recipes.jl")


function __init__()
    _registerrecipes()
    foreach(values(_tasks)) do t
        if !haskey(learningtasks(), t.id)
            push!(learningtasks(), t)
        end
    end
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        import .Makie
        import .Makie: @recipe, @lift
        import .FastAI: ShowMakie
        include("makie.jl")
    end
end

export TableRow, TabularPreprocessing, TabularClassificationSingle, TabularRegression

end
