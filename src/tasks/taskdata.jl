# Helpers for creating data containers and iterators over
# encoded obseravations.

# ## Data container

"""
    taskdataset(data, task, context)

Transform data container `data` of samples into a data container
of encoded samples.
Maps `encodesample(task, context, sample)` over the observations in
`data`. Also handles in-place `MLUtils.getobs!` through `encodesample!`.
"""
struct TaskDataset{TData, TTask<:LearningTask, TContext<:Context}
    data::TData
    task::TTask
    context::TContext
end

Base.length(ds::TaskDataset) = numobs(ds.data)

function Base.getindex(ds::TaskDataset, idx)
    return encodesample(ds.task, ds.context, getobs(ds.data, idx))
end

function MLUtils.getobs!(buf, ds::TaskDataset, idx)
    return encodesample!(buf, ds.task, ds.context, getobs(ds.data, idx))
end


"""
    taskdataset(data, task, context)

Transform data container `data` of samples into a data container of `(x, y)`-pairs.
Maps `encodesample(task, context, sample)` over the observations in `data`.
"""
const taskdataset = TaskDataset


# ## Data iterator


"""
    taskdataloaders(data, task[, batchsize])
    taskdataloaders(traindata, validdata, task[, batchsize; shuffle = true, dlkwargs...])

Create training and validation `DataLoader`s from two data containers
`(traindata, valdata)`. If only one container `data` is passed, splits
it into two, with `pctgvalid`% of the data going into the validation split.

## Arguments

Positional:
- `batchsize = 16`

Keyword:
- `shuffle = true`: Whether to shuffle the training data container
- `validbsfactor = 2`: Factor to multiply batchsize for validation data loader with (validation
    batches can be larger since no GPU memory is needed for the backward pass)

All remaining keyword arguments are passed to [`DataLoader`](#).

## Examples

Basic usage:

```julia
traindl, validdl = taskdataloaders(data, task, 128)
```

Explicit validation data container and no shuffling of training container:

```julia
traindl, validdl = taskdataloaders(traindata, validdata, task, shuffle=false)
```

Customizing the [`DataLoader`](#)

```julia
traindl, validdl = taskdataloaders(data, task, parallel=false, buffered=false)
```
"""
function taskdataloaders(
        traindata,
        validdata,
        task::LearningTask,
        batchsize = 16;
        shuffle = true,
        validbsfactor = 2,
        parallel = true,
        collate = true,
        kwargs...)
    return (
        DataLoader(taskdataset(traindata, task, Training()); batchsize, shuffle, collate, parallel, kwargs...),
        DataLoader(taskdataset(validdata, task, Validation());
                   batchsize = validbsfactor * batchsize, collate, parallel, kwargs...),
    )
end


function taskdataloaders(
        data,
        task::LearningTask,
        batchsize = 16;
        pctgval = 0.2,
        kwargs...)
    traindata, validdata = splitobs(shuffleobs(data), at = 1-pctgval)
    taskdataloaders(traindata, validdata, task, batchsize; kwargs...)
end
