module FastText

using FastAI
using FastAI:
              Datasets,
# blocks
              Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
              LabelMulti, wrapped, Continuous, getencodings, getblocks, encodetarget,
              encodeinput,
# encodings
              Encoding, StatefulEncoding, OneHot,
# visualization
              ShowText,
# other
              Context, Training, Validation

using FastAI.Datasets

using InlineTest
using Random
using TextAnalysis:
    StringDocument, prepare!, strip_stopwords,
    strip_html_tags, strip_non_letters, strip_numbers
using Tables
using ShowCases


include("recipes.jl")
include("blocks/text.jl")
include("transform.jl")
include("encodings/textpreprocessing.jl")

const _tasks = Dict{String,Any}()
include("tasks/classification.jl")

const DEFAULT_SANITIZERS = [
    replace_all_caps,
    replace_sentence_case,
    convert_lowercase,
    remove_punctuations,
    basic_preprocessing,
    remove_extraspaces
]

const DEFAULT_TOKENIZERS = [tokenize]

function __init__()
    FastAI.Registries.registerrecipes(@__MODULE__, RECIPES)
    foreach(values(_tasks)) do t
        if !haskey(FastAI.learningtasks(), t.id)
            push!(FastAI.learningtasks(), t)
        end
    end
end

export Paragraph, TextClassificationSingle, Sanitize, Tokenize

end
