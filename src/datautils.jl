
"""
methoddataloaders((traindata, validdata), method[; batchsize, dlkwargs...])
methoddataloaders(data, method[; pctgvalid, batchsize, dlkwargs])

Create training and validation `DataLoader`s from two data containers `(traindata, valdata)`.
If only one container `data` is passed, splits it into two with `pctgvalid`% of the data
going into the validation split.

Other keyword arguments are passed to `DataLoader`s.
"""
function methoddataloaders(datas::NTuple{2}, method, batchsize = 16; kwargs...)
    traindata, validdata = datas
return (
    DataLoader(shuffleobs(methoddataset(traindata, method, Training())), batchsize; kwargs...),
    DataLoader(methoddataset(validdata, method, Validation()), batchsize; kwargs...),
)
end

function methoddataloaders(
        data,
        method,
        batchsize = 16;
        pctgval = 0.2,
        shuffle = true,
        kwargs...)
    data = shuffle ? shuffleobs(data) : data
    methoddataloaders(splitobs(data, at = 1-pctgval), method, batchsize; kwargs...)
end
