"""
    TextEncoding() <: Encoding

Encodes `Paragraph`s by applying various textual transforms.


Encodes
- `Paragraph` -> `Paragraph`

"""
struct Sanitize <: Encoding
    tfms
end

Sanitize() = Sanitize(DEFAULT_SANITIZERS)


encodedblock(::Sanitize, block::Paragraph) = block

function encode(p::Sanitize, context, block::Paragraph, obs)
    for tfm in values(p.tfms)
        obs = tfm(obs)
    end
    obs
end

struct Tokenize <: Encoding
    tfms
end

Tokenize() = Tokenize(DEFAULT_TOKENIZERS)

function encodedblock(p::Tokenize, block::Paragraph)
    return Tokens()
end

function encode(p::Tokenize, context, block::Paragraph, obs)
    for tfm in values(p.tfms)
        obs = tfm(obs)
    end
    obs
end

function computevocabulary(data; vocab_size=40000)
    lookup_table = Dict{String, Int}()

    enc1 = Sanitize()
    sanitized_Data = map(i -> encode(enc1, Training(), Paragraph(), getobs(data, i)[1]), 1:numobs(data))

    enc2 = Tokenize()
    tokenized_data = map(i -> encode(enc2, Training(), Paragraph(), getobs(sanitized_Data, i)), 1:numobs(data))

    vocab = []
    for sample in tokenized_data
        for token in sample
            lookup_table[token] = get(lookup_table, token, 0) + 1
        end
    end

    ordered_dict = sort(OrderedDict(lookup_table), byvalue=true, rev=true)

    vocab = []

    for (key, value) in ordered_dict
        if vocab_size >= length(vocab)
            push!(vocab, key)
        end
    end

    return vocab

end

struct EmbedVocabulary <: Encoding
    vocab
end

function EmbedVocabulary(; vocab)
    return EmbedVocabulary(vocab)
end

function setup(::Type{EmbedVocabulary}, data; vocab_size=238483)
    vocab = computevocabulary(data, vocab_size=vocab_size)
    return EmbedVocabulary(vocab = vocab)
end

function encodedblock(p::EmbedVocabulary, block::Tokens)
    return NumberVector()
end

function encode(p::EmbedVocabulary, context, block::Tokens, obs)
    vocabulary = p.vocab

    return [indexin(vocabulary, obs) for token in obs]
end


# ## Tests

@testset "TextPreprocessing [Encoding]" begin
    sample_input = "Unsanintized text, this has to be sanitized. Then it should be tokenized. Finally it has to be numericalized"
    block = Paragraph()
    enc1 = Sanitize()
    testencoding(enc1, block, sample_input)

    # sample_input_sanitized = "xxbos xxmaj unsanintized text sanitized xxmaj tokenized xxmaj finally numericalized"
    sample_input_sanitized = encode(enc1, Training(), block, sample_input)
    block = Paragraph()
    enc2 = Tokenize()
    testencoding(enc2, block, sample_input_sanitized)

    # tokenized_input = ["xxbos", "xxmaj", "unsanintized", "text", "sanitized", "tokenized", "finally", "numericalized"]
    tokenized_input = encode(enc2, Training(), block, sample_input_sanitized)
    block = Tokens()
    vocab = setup(EmbedVocabulary, [[sample_input]])
    enc3 = EmbedVocabulary(vocab = vocab.vocab)
    testencoding(enc3, block, tokenized_input)


end
