function blockmodel(inblock::NumberVector, outblock::OneHotTensor, backbone; k = 10)

    classifier = TextClassifier(backbone)
    return (input) -> model(input, k = k, classifier = classifier)
end