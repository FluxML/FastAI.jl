module Textual


using ..FastAI
using ..FastAI:
    Datasets,
    # blocks
    Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
    LabelMulti, wrapped, Continuous, getencodings, getblocks, encodetarget, encodeinput,
    # encodings
    Encoding, StatefulEncoding, OneHot,
    # visualization
    ShowText,
    # other
    Context, Training, Validation
using ..FastAI.Datasets

using ..FastAI.Datasets

# for tests
using ..FastAI: testencoding

# extending
import ..FastAI:
    blockmodel, blockbackbone, blocklossfn, encode, decode, checkblock,
    encodedblock, decodedblock, showblock!, mockblock, setup, encodestate,
    decodestate



import Requires: @require

using InlineTest
using Random
using TextAnalysis
using DataStructures

using WordTokenizers: TokenBuffer, isdone, character, spaces, nltk_url1, nltk_url2, nltk_phonenumbers

include("recipes.jl")
include("blocks/text.jl")
include("transform.jl")
include("encodings/textpreprocessing.jl")

const _tasks = Dict{String,Any}()
include("tasks/classification.jl")


function __init__()
    _registerrecipes()
    foreach(values(_tasks)) do t
        if !haskey(FastAI.learningtasks(), t.id)
            push!(FastAI.learningtasks(), t)
        end
    end
end

export Paragraph, TokenVector, TextClassficationSingle, Sanitize, Tokenize, replace_all_caps, replace_sentence_case, convert_lowercase
end
