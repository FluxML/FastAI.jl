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
struct LanguageModel{A,F}
    vocab::A
    layers::F
end

function LanguageModel(load_pretrained::Bool = false, task::Any = Nothing; embedding_size::Integer = 400, hid_lstm_sz::Integer = 1150,
    out_lstm_sz::Integer = embedding_size, embed_drop_prob::Float32 = 0.05f0, in_drop_prob::Float32 = 0.4f0, hid_drop_prob::Float32 = 0.5f0,
    layer_drop_prob::Float32 = 0.3f0, final_drop_prob::Float32 = 0.3f0)
    vocab = task.encodings[3].vocab.keys
    de = DroppedEmbeddings(length(vocab), embedding_size, embed_drop_prob; init = (dims...) -> init_weights(0.1, dims...))
    lm = LanguageModel(
        vocab,
        Chain(
            de,
            VarDrop(in_drop_prob),
            WeightDroppedLSTM(embedding_size, hid_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1 / hid_lstm_sz, dims...)),
            VarDrop(layer_drop_prob),
            WeightDroppedLSTM(hid_lstm_sz, hid_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1 / hid_lstm_sz, dims...)),
            VarDrop(layer_drop_prob),
            WeightDroppedLSTM(hid_lstm_sz, out_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1 / hid_lstm_sz, dims...)),
            VarDrop(final_drop_prob),
            Base.Fix2(de, true),
            softmax
        )
    )
    # load_pretrained && load_model!(lm, datadep"Pretrained ULMFiT Language Model/ulmfit_lm_en.bson")
    return lm
end

Flux.@functor LanguageModel
Flux.trainable(m::LanguageModel) = (layers = m.layers)

function loss(m::LanguageModel, xs, y; k = 10)
    # forward steps
    # reset!(m.layers)
    # Zygote.ignore() do
    #     [m.layers(x) for x in xs[1:(end - k)]]
    # end
    # bptt
    ypreds = [m.layers(x) for x in xs]
    l = sum(Flux.Losses.logitcrossentropy.(ypreds, y))
    println("Loss: $loss")
    return ypreds
end

function train_language_model(lm::LanguageModel = Nothing, batches = Nothing)
    opt = Adam(1e-4)

    for batch in batches
        xs, y = batch
        ps = Flux.params(lm)
        gs = gradient(() -> loss(lm, batch...), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end

end