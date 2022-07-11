"""
    predict(task, model, input[; device, context])

Predict a `target` from `input` using `model`. Optionally apply function `device`
to `x` before passing to `model` and use `context` instead of the default
context [`Inference`](#).
"""
function predict(task, model, input; device = cpu, undevice = cpu, context = Inference())
    if shouldbatch(task)
        return predictbatch(task,
                            model,
                            [input];
                            device = device,
                            undevice = undevice,
                            context = context) |> only
    else
        return decodeypred(task,
                           context,
                           undevice(model(device(encodeinput(task, context, input)))))
    end
end

"""
    predictbatch(task, model, inputs[; device, context])

Predict `targets` from a vector of `inputs` using `model` by batching them.
Optionally apply function `device` to batch before passing to `model` and
use `context` instead of the default [`Inference`](#).
"""
function predictbatch(task,
                      model,
                      inputs;
                      device = cpu,
                      undevice = cpu,
                      context = Inference())
    xs = device(MLUtils.batch([copy(encodeinput(task, context, input)) for input in inputs]))
    ŷs = undevice(model(xs))
    targets = [decodeypred(task, context, ŷ) for ŷ in Datasets.unbatch(ŷs)]
    return targets
end
