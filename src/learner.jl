
"""
    methodlearner(method, data, model[; kwargs...]) -> Learner

Create a `Learner` to train a model for learning method `method` using
`data`. See also [`Learner`](#).

## Keyword arguments

- `isbackbone = true`: Whether `model` is a backbone or a complete model. If `true`,
    a model will be constructed using [`methodmodel`](#)`(method, model)`.
- `validdata = nothing`: Validation data container. If none is given, `data` will be
    randomly split into training and validation data.
- `optimizer = ADAM()`: Optimizer passed to `Learner`.
- `lossfn = `[`methodlossfn`](#)`(method)`: Loss function passed to `Learner`.

Any other keyword arguments will be passed to [`methoddataloaders`](#).
"""
function methodlearner(
        method::LearningMethod,
        data,
        backbone,
        callbacks...;
        isbackbone=true,
        pctgval=0.2,
        batchsize=16,
        validdata=nothing,
        validbsfactor=2,
        optimizer=ADAM(),
        lossfn=methodlossfn(method),
        dlkwargs=(;),
    )
    model = isbackbone ? methodmodel(method, backbone) : backbone
    dls = if isnothing(validdata)
        methoddataloaders(data, method, batchsize;
            validbsfactor=validbsfactor, pctgval=pctgval, dlkwargs...)
    else
        methoddataloaders(data, validdata, method, batchsize;
            validbsfactor=validbsfactor, dlkwargs...)
    end
    return Learner(model, dls, optimizer, lossfn, callbacks...)
end
