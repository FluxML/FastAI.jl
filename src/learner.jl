
"""
    methodlearner(method, data, backbone; <kwargs>...) -> Learner

Create a `Learner` to train a model for learning method `method` using
`data`. See also [`Learner`](#).
"""
function methodlearner(
        method::LearningMethod,
        data,
        backbone,
        callbacks...;
        isbackbone = true,
        pctgval = 0.2,
        batchsize = 16,
        validdata = nothing,
        validbsfactor = 2,
        optimizer = ADAM(),
        lossfn = methodlossfn(method),
        dlkwargs = (;),
    )
    model = isbackbone ? methodmodel(method, backbone) : backbone
    dls = if isnothing(validdata)
        methoddataloaders(data, method, batchsize;
            validbsfactor = validbsfactor, pctgval = pctgval)
    else
        methoddataloaders(data, validdata, method, batchsize;
            validbsfactor = validbsfactor)
    end
    return Learner(model, dls, optimizer, lossfn, callbacks...)
end
