function blockmodel(inblock::NumberVector, outblock::OneHotTensor, backbone; k = 10)

    classifier = TextClassifier(backbone)
    return classifier
end

function blockmodel(inblock::NumberVector, outblock::NumberVector, backbone; k = 10)
    return backbone
end

function (b::TextClassifier)(input)
    k = 10
    Zygote.ignore() do
        Flux.reset!(b.rnn_layers)
        [b.rnn_layers(x) for x in input[1:(end - k)]]
    end

    # bptt
    model = b.linear_layers([b.rnn_layers(x) for x in input[(end - k + 1):end]])
end

function (b::LanguageModel)(input)
    # bptt
    model = [b.layers(x) for x in input[1:end]]
end