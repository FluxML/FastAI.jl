module Tabular


using ..FastAI
using ..FastAI:
    # blocks
    Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
    LabelMulti, wrapped, Continuous,
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
import Flux: Embedding, Chain, Dropout, Dense, Parallel
import PrettyTables
import Requires: @require
import ShowCases: ShowCase
import Tables
import Statistics


# Blocks
include("blocks/tablerow.jl")

# Encodings
include("encodings/tabularpreprocessing.jl")


include("models.jl")
include("learningmethods.jl")
include("recipes.jl")


function __init__()
    _registerrecipes()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        import .Makie
        import .Makie: @recipe, @lift
        import .FastAI: ShowMakie
        include("makie.jl")
    end
end

export TableRow, TabularPreprocessing, TabularClassificationSingle, TabularRegression

end
