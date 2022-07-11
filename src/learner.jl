
"""
    tasklearner(task, traindata, validdata[; callbacks=[], kwargs...]) -> Learner
    tasklearner(task, data; pctgval = 0.2, kwargs...)

Create a [`Learner`](#) to train a model for learning task `task` using
`data`.

## Keyword arguments

- `callbacks = []`: [`Callback`](#)s to use during training.
- `batchsize = 16`: Batch size used for the training data loader.
- `backbone = nothing`: Backbone model to construct task-specific model from using
   [`taskmodel`](#)`(task, backbone)`.
- `model = nothing`: Complete model to use. If given, the `backbone` argument is ignored.
- `optimizer = Adam()`: Optimizer passed to `Learner`.
- `lossfn = `[`tasklossfn`](#)`(task)`: Loss function passed to `Learner`.

Any other keyword arguments will be passed to [`taskdataloaders`](#).

## Examples

Full example:

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
task = ImageClassificationSingle(blocks)
learner = tasklearner(task, data)
fitonecycle!(learner, 10)
```

Custom training and validation split:

```julia
learner = tasklearner(task, traindata, validdata)
```

Using callbacks:

```julia
learner = tasklearner(task, data; callbacks=[
    ToGPU(), Checkpointer(), LogMetrics(TensorboardBackend())
])
```
"""
function tasklearner(task::LearningTask,
                     traindata,
                     validdata;
                     backbone = nothing,
                     model = nothing,
                     callbacks = [],
                     pctgval = 0.2,
                     batchsize = 16,
                     optimizer = Adam(),
                     lossfn = tasklossfn(task),
                     kwargs...)
    if isnothing(model)
        model = isnothing(backbone) ? taskmodel(task) : taskmodel(task, backbone)
    end
    dls = taskdataloaders(traindata, validdata, task, batchsize; kwargs...)
    return Learner(model, dls, optimizer, lossfn, callbacks...)
end

function tasklearner(task, data; pctgval = 0.2, kwargs...)
    traindata, validdata = splitobs(data, at = 1 - pctgval)
    return tasklearner(task, traindata, validdata; kwargs...)
end

"""
    getbatch(learner[; validation = false, n = nothing])

Get a batch of data from `learner`. Take a batch of training data by default
or validation data if `validation = true`. If `n` take only the first
`n` samples from the batch.

"""
function getbatch(learner; context = Training(), n = nothing)
    dl = context == Validation() ? learner.data.validation : learner.data.training
    batch = first(dl)
    b = min(isnothing(n) ? Inf : n, Datasets.batchsize(batch))
    batch = MLUtils.batch([s for (s, _) in zip(Datasets.unbatch(batch), 1:b)])
    return batch
end

# ## Tests

@testset "getbatch" begin
    batch = rand(1, 10), rand(1, 10)
    learner = Learner(identity, ([batch], [batch]), nothing, nothing)
    @test size.(getbatch(learner)) == ((1, 10), (1, 10))
    @test size.(getbatch(learner, n = 4)) == ((1, 4), (1, 4))
end

@testset "tasklearner" begin
    task = SupervisedTask((Label(1:2), Label(1:2)), (OneHot(),))
    data = (rand(1:2, 1000), rand(1:2, 1000))
    @test_nowarn learner = tasklearner(task, data, model = identity)

    @testset "batch sizes" begin
        learner = tasklearner(task, data, model = identity, batchsize = 100)
        @test length(learner.data.training) == 8
        @test length(learner.data.validation) == 1

        learner = tasklearner(task, data, model = identity, pctgval = 0.4, batchsize = 100)
        @test length(learner.data.training) == 6
        @test length(learner.data.validation) == 2

        learner = tasklearner(task, data, model = identity, batchsize = 100,
                              validbsfactor = 1)
        @test length(learner.data.training) == 8
        @test length(learner.data.validation) == 2
    end

    @testset "callbacks" begin
        learner = tasklearner(task, data, model = identity,
                              callbacks = [ToGPU(), Checkpointer(mktempdir())])
        @test !isnothing(FluxTraining.getcallback(learner, Checkpointer))
    end
end
