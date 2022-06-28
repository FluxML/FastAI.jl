"""
    TextEncoding() <: Encoding

Encodes `Paragraph`s by applying various textual transforms.


Encodes
- `Paragraph` -> `Paragraph`

"""
struct Sanitize <: Encoding
    tfms
end

function Sanitize()
    base_tfms = [
        replace_all_caps,
        replace_sentence_case,
        convert_lowercase,
        remove_punctuations,
        basic_preprocessing,
        remove_extraspaces,
    ]
    return Sanitize(base_tfms)
end

function encodedblock(p::Sanitize, block::Paragraph)
    return block
end

function encode(p::Sanitize, context, block::Paragraph, obs)
    for tfm in values(p.tfms)
        obs = tfm(obs)
    end
    obs
end

struct Tokenize <: Encoding
    tfms
end

function Tokenize()
    base_tfms = [
        tokenize,
    ]
    return Tokenize(base_tfms)
end

function encodedblock(p::Tokenize, block::Paragraph)
    return TokenVector()
end

function encode(p::Tokenize, context, block::Paragraph, obs)
    for tfm in values(p.tfms)
        obs = tfm(obs)
    end
    obs
end

function computevocabulary(data)
    lookup_table = Dict()

    enc1 = Sanitize()
    sanitized_Data = mapobs((i) -> encode(enc1, Training(), Paragraph(), getobs(data, i)[1]), 1:25000)

    enc2 = Tokenize()
    tokenized_data = mapobs((i) -> encode(enc2, Training(), Paragraph(), getobs(sanitized_Data, i)), 1:25000)

    vocab = []
    for sample in tokenized_data
        for token in sample
            lookup_table[token] = get(lookup_table, token, 0) + 1
        end
    end
    return OrderedDict(lookup_table)
end

struct EmbedVocabulary <: Encoding
    vocab
end

function EmbedVocabulary(; vocab)
    return EmbedVocabulary(vocab)
end

function setup(::Type{EmbedVocabulary}, data)
    vocab = computevocabulary(data)
    return EmbedVocabulary(vocab)
end

function encodedblock(p::EmbedVocabulary, block::TokenVector)
    return NumberVector()
end

function encode(p::EmbedVocabulary, context, block::TokenVector, obs)
    vocabulary = p.vocab

    return_vect = []

    for token in obs
        push!(return_vect, getindex(vocabulary, token))
    end
    return return_vect
end



# function encode(p::Numericalize, context, block::Paragraph, obs)
#     ordered_dict = OrderedDict(p.vocab)

#     p = TextEncoding()
#     # tokenized_sample = encode(enc, Training(), Paragraph(), obs)
#     for tfm in values(p.tfms)
#         obs = tfm(obs)
#     end
#     tokenized_text = obs

#     return_vect = []

#     for token in tokenized_sample
#         push!(return_vect, getindex(ordered_dict, token))
#     end
#     return_vect
# end