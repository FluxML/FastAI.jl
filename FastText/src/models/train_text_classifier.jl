"""
ULMFiT - Text Classifier

This is wrapper around the LanguageMode struct. It has three fields:

vocab           : contains the same vocabulary from the LanguageModel
rnn_layers      : contains same DroppedEmebeddings, LSTM (AWD_LSTM) and VarDrop layers of LanguageModel except for last softmax layer
linear_layers   : contains Chain of two Dense layers [PooledDense and Dense] with softmax layer

To train create and instance and give it as first argument to 'train_classifier!' function
"""
mutable struct TextClassifier
    vocab::Vector
    rnn_layers::Flux.Chain
    linear_layers::Flux.Chain
end

function TextClassifier(lm::LanguageModel=LanguageModel(), clsfr_out_sz::Integer=2, clsfr_hidden_sz::Integer=50, clsfr_hidden_drop::Float64=0.4)
    return TextClassifier(
        lm.vocab,
        lm.layers[1:8],
        Chain(
            PooledDense(length(lm.layers[7].layer.cell.h), clsfr_hidden_sz),
            BatchNorm(clsfr_hidden_sz, relu),
            Dropout(clsfr_hidden_drop),
            Dense(clsfr_hidden_sz, clsfr_out_sz),
            BatchNorm(clsfr_out_sz),
            softmax
        )
    )
end

Flux.@functor TextClassifier

function loss(m, xs, y; k = 10)
    # forward steps
    # Flux.reset!(m.rnn_layers)
    Zygote.ignore() do
        [m.rnn_layers(x) for x in xs[1:(end - k)]]
    end
    # bptt
    ypreds = m.linear_layers([m.rnn_layers(x) for x in xs[(end - k + 1):end]])
    loss = Flux.Losses.logitcrossentropy(ypreds, y)
    println("Loss: $loss")
    return loss
end

function train_text_classifier(classifier::TextClassifier = Nothing, batches = Nothing)
    opt = Adam(1e-4)

    for batch in batches
        xs, y = batch
        ps = Flux.params(classifier)
        gs = gradient(() -> loss(classifier, batch...), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end

end
