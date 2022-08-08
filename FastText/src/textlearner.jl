# function taskmodel(task; k = 10)
#     backbone = LanguageModel(false, task)
#     return blockmodel(getblocks(task).x, getblocks(task).ŷ, backbone, k = k)
# end

function model(input; k = 10, classifier = Nothing)

    Zygote.ignore() do
        [classifier.rnn_layers(x) for x in input[1:(end - k)]]
    end

    # bptt
    model = classifier.linear_layers([classifier.rnn_layers(x) for x in input[(end - k + 1):end]])
end

function blockmodel(inblock::NumberVector,
                    outblock::OneHotTensor,
                    backbone; k = 10)

    classifier = TextClassifier(backbone)

    return (input) -> model(input, k = k, classifier = classifier)
end