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
            gpu(PooledDense(length(lm.layers[7].layer.cell.h), clsfr_hidden_sz)),
            gpu(BatchNorm(clsfr_hidden_sz, relu)),
            Dropout(clsfr_hidden_drop),
            gpu(Dense(clsfr_hidden_sz, clsfr_out_sz)),
            gpu(BatchNorm(clsfr_out_sz)),
            softmax
        )
    )
end

Flux.@functor TextClassifier

"""
Cross Validate

This function will be used to cross-validate the classifier

Arguments:

tc              : Instance of TextClassfier
gen             : 'Channel' to get a mini-batch from validation set
num_of_batches  : specifies the number of batches the validation will be done

If num_of_batches is not specified then all the batches which can be given by the
gen will be used for validation
"""
function validate(tc::TextClassifier, gen::Channel, num_of_batches::Union{Colon, Integer})
    n_classes = size(tc.linear_layers[end-2].W, 1)
    classifier = tc
    Flux.testmode!(classifier)
    loss = 0
    iters = take!(gen)
    ((num_of_batches != :) & (num_of_batches < iters)) && (iters = num_of_batches)
    TP, TN = gpu(zeros(Float32, n_classes, 1)), gpu(zeros(Float32, n_classes, 1))
    FP, FN = gpu(zeros(Float32, n_classes, 1)), gpu(zeros(Float32, n_classes, 1))
    for i=1:num_of_batches
        X = take!(gen)
        Y = gpu(take!(gen))
        X = map(x -> indices(x, classifier.vocab, "_unk_"), X)
        H = classifier.rnn_layers.(X)
        H = classifier.linear_layers(H)
        l = crossentropy(H, Y)
        Flux.reset!(classifier.rnn_layers)
        TP .+= sum(H .* Y, dims=2)
        FN .+= sum(((-1 .* H) .+ 1) .* Y, dims=2)
        FP .+= sum(H .* ((-1 .* Y) .+ 1), dims=2)
        TN .+= sum(((-1 .* H) .+ 1) .* ((-1 .* Y) .+ 1), dims=2)
        loss += l
    end
    precisions = TP ./ (TP .+ FP)
    recalls = TP ./ (TP .+ FN)
    F1 = (2 .* (precisions .* recalls)) ./ (precisions .+ recalls)
    accuracy = (TP[1] + TN[1])/(TP[1] + TN[1] + FP[1] + FN[1])
    return (loss, accuracy, precisions, recalls, F1)
end

"""
Forward pass

This funciton does the main computation of a mini-batch.
It computes the output of the all the layers [RNN and DENSE layers] and returns the predicted output for that pass.
It uses Truncated Backprop through time to compute the output.

Arguments:
tc              : Instance of TextClassifier
gen             : data loader, which will give 'X' of the mini-batch in one call
tracked_steps   : This is the number of tracked time-steps for Truncated Backprop thorugh time,
                  these will be last time-steps for which gradients will be calculated.
"""
function forward(tc::TextClassifier, gen::Channel, tracked_steps::Integer=32)
  	# swiching off tracking
    classifier = tc
    X = take!(gen)
    # println("X = $X")
    l = length(X)
    # Truncated Backprop through time
    println("l = $l")
    Zygote.ignore() do
	for i=1:ceil(l/tracked_steps)-1   # Tracking is swiched off inside this loop
        println("i = $i / $(ceil(l/tracked_steps)-1)")
	    (i == 1 && l%tracked_steps != 0) ? (last_idx = l%tracked_steps) : (last_idx = tracked_steps)
	    H = broadcast(x -> indices(x, classifier.vocab, "_unk_"), X[1:last_idx])
	    H = classifier.rnn_layers.(H)
	    X = X[last_idx+1:end]
	end

    println("Start shifting states")
    end
    # set the lated hidden states to original model

    for i in 2:8
        t_layer = tc.rnn_layers[i]
        unt_layer = classifier.rnn_layers[i]
        # for (t_layer, unt_layer) in zip(tc.rnn_layers[2:end], classifier.rnn_layers[2:end])
        if t_layer isa AWD_LSTM
            t_layer.layer.state = unt_layer.layer.state
            continue
        elseif !unt_layer.reset
            t_layer.mask = unt_layer.mask
            t_layer.reset = false
        end
        # end
    end
    println("End shifting")
    # last part of the sequecnes in X - Tracking is swiched on
    H = broadcast(x -> tc.rnn_layers[1](indices(x, classifier.vocab, "_unk_")), X)
    H = tc.rnn_layers[2:end].(H)
    H = tc.linear_layers(H)
    return H
end

"""
    loss(classifier::TextClassifier, gen::Channel, tracked_steps::Integer=32)

LOSS function

It takes the output of the forward funciton and returns crossentropy loss.

Arguments:

classifier    : Instance of TextClassifier
gen           : 'Channel' [data loader], to give a mini-batch
tracked_steps : specifies the number of time-steps for which tracking is on
"""
function loss(classifier::TextClassifier, gen::Channel, tracked_steps::Integer=32)
    H = forward(classifier, gen, tracked_steps)
    Y = gpu(take!(gen))
    l = crossentropy(H, Y)
    # reset!(classifier.rnn_layers)
    println("Loss = $l")
    return l
end

function discriminative_step!(layers, classifier::TextClassifier, gen::Channel, tracked_steps::Integer, ηL::Float64, opts::Vector)
    @assert length(opts) == length(layers)
    # Gradient calculation
    println("Start grads")
    grads = Zygote.gradient(() -> loss(classifier, gen, tracked_steps), get_trainable_params(layers))

    println("Done grads")
    # discriminative step
    ηl = ηL/(2.6^(length(layers)-1))
    for (layer, opt) in zip(layers, opts)
        opt.eta = ηl
        for ps in get_trainable_params([layer])
            Flux.Optimise.update!(opt, ps, grads[ps])
        end
        ηl *= 2.6
    end
    return
end

"""
    train_classifier!(classifier::TextClassifier=TextClassifier(), classes::Integer=1,
            data_loader::Channel=imdb_classifier_data, hidden_layer_size::Integer=50;kw...)

It contains main training loops for training a defined classifer for specified classes and data.
Usage is discussed in the docs.
"""
function train_classifier!(classifier::TextClassifier=TextClassifier(), classes::Integer=1, hidden_layer_size::Integer=50)

    # dala_loader = imdb_classifier_data
    stlr_cut_frac=0.1
    stlr_ratio=32
    stlr_η_max=0.01
    val_loader=nothing
    epochs=1
    checkpoint_itvl=5000
    tracked_steps=8

    trainable = []
    append!(trainable, [classifier.rnn_layers[[1, 3, 5, 7]]...])
    push!(trainable, [classifier.linear_layers[1:2]...])
    push!(trainable, [classifier.linear_layers[4:5]...])
    opts = [ADAM(0.001, (0.7, 0.99)) for i=1:length(trainable)]
    gpu(classifier.rnn_layers)

    for epoch=1:epochs
        println("Epoch: $epoch")
        gen = imdb_classifier_data(16)
        num_of_iters = take!(gen)
        cut = num_of_iters * epochs * stlr_cut_frac
        for iter=1:num_of_iters

            println("Iteration: $iter / $num_of_iters")

            # Slanted triangular learning rates
            t = iter + (epoch-1)*num_of_iters
            p_frac = (iter < cut) ? iter/cut : (1 - ((iter-cut)/(cut*(1/stlr_cut_frac-1))))
            ηL = stlr_η_max*((1+p_frac*(stlr_ratio-1))/stlr_ratio)

            # Gradual-unfreezing Step with discriminative fine-tuning
            unfreezed_layers, cur_opts = (epoch < length(trainable)) ? (trainable[end-epoch+1:end], opts[end-epoch+1:end]) : (trainable, opts)
            println("start discriminative_step")
            discriminative_step!(unfreezed_layers, classifier, gen, tracked_steps,ηL, cur_opts)

            println("End discriminative_step")

            reset_masks!.(classifier.rnn_layers)    # reset all dropout masks
        end
        println("Train set accuracy: $trn_accu , Training loss: $trn_loss")
        if val_loader != nothing
            val_loss, val_acc, val_precisions, val_reacalls, val_F1_scores = validate(classifer, val_loader)
        else
            continue
        end
        #!(val_loader isa nothing) ? (val_loss, val_acc, val_precisions, val_reacalls, val_F1_scores = validate(classifer, val_loader)) : continue
        println("Cross validation loss: $val_loss")
        println("Cross validation accuracy:\n $val_acc")
        println("Cross validation class wise Precisions:\n $val_precisions")
        println("Cross validation class wise Recalls:\n $val_recalls")
        println("Cross validation class wise F1 scores:\n $val_F1_scores")
    end
end

"""
    predict(tc::TextClassifier, text_sents::Corpus)

This function can be used to test the model after training.
It returns the predictions done by the model for given `Corpus` of `Documents`
All the preprocessing related to the used vocabulary should be done before using this function.
Use `prepare!` function to do preprocessing
"""
function predict(tc::TextClassifier, text_sents::Corpus)
    classifier = tc
    Flux.testmode!(classifier)
    predictions = []
    expr(x) = indices(x, classifier.vocab, "_unk_")
    for text in text_sents
        tokens_ = tokens(text)
        h = classifier.rnn_layers.(expr.(tokens_))
        probability_dist = classifier.linear_layers(h)
        class = argmax(probability_dist)
        push!(predictions, class)
    end
    return predictions
end
