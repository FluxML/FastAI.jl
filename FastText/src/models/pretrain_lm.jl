using DelimitedFiles

"""
ULMFiT - LANGUAGE MODEL

    LanguageModel(load_pretrained::Bool=false, vocabpath::String="vocabs/lm_vocab.csv";kw...)

The Language model structure for ULMFit is defined by 'LanguageModel' struct.
It contains has two fields:
    vocab   : vocabulary, which will be used for language modelling
    layers  : embedding, RNN and dropout layers of the whole model
In this language model, the embedding matrix used in the embedding layer
is same for the softmax layer, following Weight-tying technique.
The field 'layers' also includes the Variational Dropout layers.
It takes several dropout probabilities for different dropout for different layers.

[Usage and arguments are discussed in the docs]
There are several keyword argunments to set the dropout probabilities
of the layers of model checkout those arguments in the docs.
# Example:

julia> lm = LanguageModel()
"""
mutable struct LanguageModel
    vocab :: Vector
    layers :: Flux.Chain
end

function LanguageModel(load_pretrained::Bool=false, task::Any = Nothing;embedding_size::Integer=400, hid_lstm_sz::Integer=1150, out_lstm_sz::Integer=embedding_size,
    embed_drop_prob::Float64 = 0.05, in_drop_prob::Float64 = 0.4, hid_drop_prob::Float64 = 0.5, layer_drop_prob::Float64 = 0.3, final_drop_prob::Float64 = 0.3)
    vocab = task.encodings[3].vocab.keys
    de = DroppedEmbeddings(length(vocab), embedding_size, embed_drop_prob; init = (dims...) -> init_weights(0.1, dims...))
    lm = LanguageModel(
        vocab,
        Chain(
            de,
            VarDrop(in_drop_prob),
            AWD_LSTM(embedding_size, hid_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1/hid_lstm_sz, dims...)),
            VarDrop(layer_drop_prob),
            AWD_LSTM(hid_lstm_sz, hid_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1/hid_lstm_sz, dims...)),
            VarDrop(layer_drop_prob),
            AWD_LSTM(hid_lstm_sz, out_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1/hid_lstm_sz, dims...)),
            VarDrop(final_drop_prob),
            x -> de(x, true),
            softmax
        )
    )
    # load_pretrained && load_model!(lm, datadep"Pretrained ULMFiT Language Model/ulmfit_lm_en.bson")
    return lm
end

Flux.@functor LanguageModel
