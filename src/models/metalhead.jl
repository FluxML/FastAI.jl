

function resnet50(;pretrained = true, keephead = false)
    model = Metalhead.resnet50()
    if !pretrained
        return model
    end

    weights = BSON.load(datadep"weights-resnet50/resnet50.bson")[:weights]
    Flux.loadparams!(model, weights)

    if !keephead
        model = model[1:end-3]
    end

    return model
end
