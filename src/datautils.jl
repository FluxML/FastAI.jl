
"""
    methoddataloaders(data, method)
    methoddataloaders(traindata, validdata, method[, batchsize; shuffle = true, dlkwargs...])

Create training and validation `DataLoader`s from two data containers `(traindata, valdata)`.
If only one container `data` is passed, splits it into two with `pctgvalid`% of the data
going into the validation split.

Other keyword arguments are passed to `DataLoader`s.
"""
function methoddataloaders(
        traindata,
        validdata,
        method::LearningMethod,
        batchsize = 16;
        shuffle = true,
        kwargs...)
    traindata = shuffle ? shuffleobs(traindata) : traindata
    return (
        DataLoader(methoddataset(traindata, method, Training()), batchsize; kwargs...),
        DataLoader(methoddataset(validdata, method, Validation()), batchsize; kwargs...),
    )
end


function methoddataloaders(
        data,
        method::LearningMethod,
        batchsize = 16;
        pctgval = 0.2,
        shuffle = true,
        kwargs...)
    data = shuffle ? shuffleobs(data) : data
    traindata, validdata = splitobs(data, at = 1-pctgval)
    methoddataloaders(traindata, validdata, method, batchsize; shuffle = false, kwargs...)
end
