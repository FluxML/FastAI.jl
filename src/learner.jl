
"""
    methodlearner(method, data, backbone; <kwargs>...) -> Learner

Create a `Learner` to train a model for learning method `method` using
`data`. See also [`Learner`](#).
"""
function methodlearner(
        method,
        data,
        backbone,
        callbacks...;
        isbackbone = true,
        pctgval = 0.2,
        batchsize = 16,
        validbsfactor = 2,
        optimizer = ADAM(),
        lossfn = methodlossfn(method),
        dlkwargs = (;),
    )
    model = isbackbone ? methodmodel(method, backbone) : backbone
    dls = methoddataloaders(data;
        batchsize = batchsize, validbsfactor = validbsfactor, pctgval = pctgval)
    return Learner(model, dls, optimizer, lossfn, callbacks...)
end
