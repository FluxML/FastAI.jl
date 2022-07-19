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

function LanguageModel(load_pretrained::Bool=false, vocabpath::String=joinpath(@__DIR__,"vocabs/lm_vocab.csv");embedding_size::Integer=400, hid_lstm_sz::Integer=1150, out_lstm_sz::Integer=embedding_size,
    embed_drop_prob::Float64 = 0.05, in_drop_prob::Float64 = 0.4, hid_drop_prob::Float64 = 0.5, layer_drop_prob::Float64 = 0.3, final_drop_prob::Float64 = 0.3)
    vocab = (string.(readdlm(vocabpath, ',')))[:, 1]
    de = gpu(DroppedEmbeddings(length(vocab), embedding_size, embed_drop_prob; init = (dims...) -> init_weights(0.1, dims...)))
    lm = LanguageModel(
        vocab,
        Chain(
            de,
            VarDrop(in_drop_prob),
            gpu(AWD_LSTM(embedding_size, hid_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1/hid_lstm_sz, dims...))),
            VarDrop(layer_drop_prob),
            gpu(AWD_LSTM(hid_lstm_sz, hid_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1/hid_lstm_sz, dims...))),
            VarDrop(layer_drop_prob),
            gpu(AWD_LSTM(hid_lstm_sz, out_lstm_sz, hid_drop_prob; init = (dims...) -> init_weights(1/hid_lstm_sz, dims...))),
            VarDrop(final_drop_prob),
            x -> de(x, true),
            softmax
        )
    )
    load_pretrained && load_model!(lm, datadep"Pretrained ULMFiT Language Model/ulmfit_lm_en.bson")
    return lm
end

Flux.@functor LanguageModel

"""
    test_lm(lm::LanguageModel, data_gen, num_of_iters::Integer; unknown_token::String="_unk_")

This function is used to test the `LanguageModel` on the given data given by data_gen.
`num_of_iters` refers to number of batches for which the model has to be tested.
It returns loss, accuracy, precsion, recall and F1 score.

# Example:

julia> test_lm(lm, data_gen, 200, "<unk")
"""
function test_lm(lm::LanguageModel, data_gen, num_of_iters::Integer; unknown_token::String="_unk_")
    model_layers = lm.layers
    testmode!(model_layers)
    loss = 0
    len = length(vocab)
    TP, FP, FN, TN = zeros(len, 1), zeros(len, 1), zeros(len, 1), zeros(len, 1)
    for iter=1:num_of_iters
        X, Y = take!(gen), take!(gen)
        H = broadcast(w -> indices(w, lm.vocab, unknown_token), X)
        H = model_layers.(H)
        Y = broadcast(x -> gpu(Flux.onehotbatch(x, lm.vocab, unknown_token)), Y)
        l = sum(crossentropy.(H, Y))
        Flux.reset!(model_layers)
        for (h, y) in zip(H, Y)
            TP .+= sum(h .* y, dims=2)
            FN .+= sum(((-1 .* h) .+ 1) .* y, dims=2)
            FP .+= sum(h .* ((-1 .* y) .+ 1), dims=2)
            TN .+= sum(((-1 .* h) .+ 1) .* ((-1 .* y) .+ 1), dims=2)
        end
        loss += l/length(X[1])
    end
    precision = *((TP./(TP .+ FP))...)
    recall = *((TP./(TP .+ FN))...)
    F1 = (2*precision*recall)/(precision+recall)
    accuracy = (TP[1] + TN[1])/(TP[1] + TN[1] + FP[1] + FN[1])
    return (loss/num_of_iters), accuarcy, precision, recall, F1
end

# computes the forward pass while training
function forward(lm, batch)
    batch = map(x -> indices(x, lm.vocab, "_unk_"), batch)
    batch = gpu(batch)
    batch = lm.layers.(batch)
    return batch
end

# loss funciton - Calculates crossentropy loss
function loss(lm, gen)
    H = forward(lm, take!(gen))
    Y = broadcast(x -> gpu(Flux.onehotbatch(x, lm.vocab, "_unk_")), take!(gen))
    l = sum(Flux.crossentropy.(H, Y))
    reset!(lm.layers)
    return l
end

# Backpropagation step while training
function backward!(layers, lm, gen, opt)
    # Calulating gradients and weights updation
    p = get_trainable_params(layers)
    grads = Zygote.gradient(() -> loss(lm, gen), p)
    Flux.Optimise.update!(opt, p, grads)
    return
end

"""
pretrain_lm!

This funciton contains main training loops for pretrainin the Language model
including averaging step for the 'AWD_LSTM' layers.

Usage and arguments are explained in the docs of ULMFiT
"""
function pretrain_lm!(lm::LanguageModel=LanguageModel(), data_loader::Channel=load_wikitext_103;
    base_lr=0.004, epochs::Integer=1, checkpoint_iter::Integer=5000)

    # Initializations
    opt = ADAM(base_lr, (0.7, 0.99))    # ADAM Optimizer

    # Pre-Training loops
    for epoch=1:epochs
        println("\nEpoch: $epoch")
        gen = data_loader()
        num_of_batches = take!(gen) # Number of mini-batches
        T = num_of_iters-Int(floor((num_of_iters*2)/100))   # Averaging Trigger
        set_trigger!.(T, lm.layers)  # Setting triggers for AWD_LSTM layers
        for i=1:num_of_batches

            # REVERSE PASS
            backward!(lm.layers, lm, gen, opt)

            # ASGD Step, works after Triggering
            asgd_step!.(i, lm.layers)

            # Resets dropout masks for all the layers with Varitional DropOut or DropConnect masks
            reset_masks!.(lm.layers)

            # Saving checkpoints
            if i == checkpoint_iter save_model!(lm) end
        end
    end
end

# To save model
function save_model!(m::LanguageModel, filepath::String)
    weights = cpu.(params(m))
    BSON.@save filepath weights
end

# To load model
function load_model!(lm::LanguageModel, filepath::String)
    BSON.@load filepath weights
    # reshape saved weights to match Recurr (h, c) shape
    layers = [5, 6, 10, 11, 15, 16]
    for l in layers
        weights[l] = reshape(weights[l], length(weights[l]), 1)
    end
    Flux.loadparams!(lm, weights)
end

"""
sample(starting_text::AbstractDocument, lm::LanguageModel)

Prints sampling results taking `starting_text` as initial tokens for the sampling for LanguageModel.

# Example:

julia> sampling("computer science", lm)
SAMPLING...
, is fast growing field ......

"""
function sample(starting_text::AbstractDocument, lm::LanguageModel; out_size = 100)
    testmode!(lm.layers)
    model_layers = lm.layers
    tokens = TextAnalysis.tokens(starting_text)
    word_indices = map(x -> indices([x], lm.vocab, "_unk_"), tokens)
    h = (model_layers.(word_indices))[end]
    prediction = lm.vocab[argmax(h)[1]]
    println("SAMPLING...")
    print(prediction, ' ')
    curr_count = 1
    while true

        h = indices([prediction], lm.vocab, "_unk_")
        h = model_layers(h)
        prediction = lm.vocab[argmax(h)[1]]
        print(prediction, ' ')
        prediction == "_pad_" && break

        curr_count += 1

        if curr_count == out_size break end
    end
end
