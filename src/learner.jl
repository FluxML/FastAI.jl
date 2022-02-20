
"""
    methodlearner(method, traindata, validdata[; callbacks=[], kwargs...]) -> Learner
    methodlearner(method, data; pctgval = 0.2, kwargs...)

Create a [`Learner`](#) to train a model for learning method `method` using
`data`.

## Keyword arguments

- `callbacks = []`: [`Callback`](#)s to use during training.
- `batchsize = 16`: Batch size used for the training data loader.
- `backbone = nothing`: Backbone model to construct task-specific model from using
   [`methodmodel`](#)`(method, backbone)`.
- `model = nothing`: Complete model to use. If given, the `backbone` argument is ignored.
- `optimizer = ADAM()`: Optimizer passed to `Learner`.
- `lossfn = `[`methodlossfn`](#)`(method)`: Loss function passed to `Learner`.

Any other keyword arguments will be passed to [`methoddataloaders`](#).

## Examples

Full example:

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
method = ImageClassificationSingle(blocks)
learner = methodlearner(method, data)
fitonecycle!(learner, 10)
```

Custom training and validation split:

```julia
learner = methodlearner(method, traindata, validdata)
```

Using callbacks:

```julia
learner = methodlearner(method, data; callbacks=[
    ToGPU(), Checkpointer(), LogMetrics(TensorboardBackend())
])
```
"""
function methodlearner(
        method::LearningMethod,
        traindata,
        validdata;
        backbone=nothing,
        model=nothing,
        callbacks=[],
        pctgval=0.2,
        batchsize=16,
        optimizer=ADAM(),
        lossfn=methodlossfn(method),
        kwargs...,
    )
    if isnothing(model)
        model = isnothing(backbone) ? methodmodel(method) : methodmodel(method, backbone)
    end
    dls = methoddataloaders(traindata, validdata, method, batchsize; kwargs...)
    return Learner(model, dls, optimizer, lossfn, callbacks...)
end

function methodlearner(method, data; pctgval=0.2, kwargs...)
    traindata, validdata = splitobs(shuffleobs(data), at=1 - pctgval)
    return methodlearner(method, traindata, validdata; kwargs...)
end


"""
    getbatch(learner[; validation = false, n = nothing])

Get a batch of data from `learner`. Take a batch of training data by default
or validation data if `validation = true`. If `n` take only the first
`n` samples from the batch.

"""
function getbatch(learner; context = Training(), n = nothing)
    dl = context==Validation() ? learner.data.validation : learner.data.training
    batch = first(learner.data.validation)
    bs = DataLoaders._batchsize(batch, DataLoaders.BatchDimLast())
    b = isnothing(n) ? bs : min(n, bs)
    batch = DataLoaders.collate([s for (s, _) in zip(DataLoaders.obsslices(batch), 1:b)])
    return batch
end

# ## Tests

@testset "getbatch" begin
    batch = rand(1, 10), rand(1, 10)
    learner = Learner(identity, ([batch], [batch]), nothing, nothing)
    @test size.(getbatch(learner)) == ((1,10), (1, 10))
    @test size.(getbatch(learner, n=4)) == ((1,4), (1,4))
end


@testset "methodlearner" begin
    method = SupervisedMethod((Label(1:2), Label(1:2)), (OneHot(),))
    data = (rand(1:2, 1000), rand(1:2, 1000))
    @test_nowarn learner = methodlearner(method, data, model=identity)

    @testset "batch sizes" begin
        learner = methodlearner(method, data, model=identity, batchsize=100)
        @test length(learner.data.training) == 8
        @test length(learner.data.validation) == 1

        learner = methodlearner(method, data, model=identity, pctgval=0.4, batchsize=100)
        @test length(learner.data.training) == 6
        @test length(learner.data.validation) == 2

        learner = methodlearner(method, data, model=identity, batchsize=100, validbsfactor=1)
        @test length(learner.data.training) == 8
        @test length(learner.data.validation) == 2
    end

    @testset "callbacks" begin
        learner = methodlearner(
            method, data, model=identity,
            callbacks=[ToGPU(), Checkpointer(mktempdir())])
        @test !isnothing(FluxTraining.getcallback(learner, Checkpointer))

    end
end
