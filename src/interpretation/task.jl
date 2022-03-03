# High-level plotting functions for use with `BlockMethod`s

"""
    showsample([backend], task, sample)

Show an unprocessed `sample` for `LearningTask` `task` to
`backend::`[`ShowBackend`](#).

## Examples

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
task = ImageClassificationSingle(data)
sample = getobs(data, 1)
showsample(task, sample)  # select backend automatically
showsample(ShowText(), task, sample)
```
"""
function showsample(backend::ShowBackend, task::AbstractBlockMethod, sample)
    blocks = ("Input" => getblocks(task)[1], "Target" => getblocks(task)[2])
    showblock(backend, blocks, sample)
end
showsample(task::AbstractBlockMethod, sample) =
    showsample(default_showbackend(), task, sample)


"""
    showsample([backend], task, sample)

Show a vector of unprocessed `samples` for `LearningTask` `task` to
`backend::`[`ShowBackend`](#).

## Examples

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
task = ImageClassificationSingle(data)
samples = [getobs(data, i) for i in 1:4]
showsamples(task, samples)  # select backend automatically
showsamples(ShowText(), task, samples)
```
"""
function showsamples(backend::ShowBackend, task::AbstractBlockMethod, samples)
    showblocks(backend, "Sample" => getblocks(task).sample, samples)
end
showsamples(task::AbstractBlockMethod, samples) =
    showsamples(default_showbackend(), task, sample)

"""
    showencodedsample([backend], task, encsample)

Show an encoded sample `encsample` to `backend`.
"""
function showencodedsample(backend::ShowBackend, task::AbstractBlockMethod, encsample)
    showblockinterpretable(
        backend,
        getencodings(task),
        getblocks(task).encodedsample,
        encsample,
    )
end
showencodedsample(task, encsample) =
    showencodedsample(default_showbackend(), task, encsample)

"""
    showencodedsamples([backend], task, encsamples)

Show a vector of encoded samples `encsamples` to `backend`.
"""
function showencodedsamples(
    backend::ShowBackend,
    task::AbstractBlockMethod,
    encsamples::AbstractVector,
)
    xblock, yblock = encodedblockfilled(getencodings(task), getblocks(task))
    showblocksinterpretable(
        backend,
        getencodings(task),
        ("x" => xblock, "y" => yblock),
        encsamples,
    )
end

"""
    showbatch([backend], task, batch)

Show a collated batch of encoded samples to `backend`.
"""
function showbatch(backend::ShowBackend, task::AbstractBlockMethod, batch)
    encsamples = collect(DataLoaders.obsslices(batch))
    showencodedsamples(backend, task, encsamples)
end
showbatch(task, batch) = showbatch(default_showbackend(), task, batch)

"""
    showprediction([backend], task, pred)
    showprediction([backend], task, sample, pred)

Show a prediction `pred`. If a `sample` is also given, show it next to
the prediction. ŷ
"""
function showprediction(backend::ShowBackend, task::AbstractBlockMethod, pred)
    showblock(backend, "Prediction" => getblocks(task).pred, pred)
end

function showprediction(backend::ShowBackend, task::AbstractBlockMethod, sample, pred)
    blocks = getblocks(task)
    showblock(
        backend,
        ("Sample" => blocks.sample, "Prediction" => blocks.pred),
        (sample, pred),
    )
end


showprediction(task::AbstractBlockMethod, args...) =
    showprediction(default_showbackend(), task, args...)

"""
    showpredictions([backend], task, preds)
    showpredictions([backend], task, samples, preds)

Show predictions `pred`. If `samples` are also given, show them next to
the prediction.
"""
function showpredictions(backend::ShowBackend, task::AbstractBlockMethod, preds)
    predblock = decodedblockfilled(getencodings(task), getblocks(task).ŷ)
    showblocks(backend, "Prediction" => predblock, preds)
end

function showpredictions(backend::ShowBackend, task::AbstractBlockMethod, samples, preds)
    predblock = decodedblockfilled(getencodings(task), getblocks(task).ŷ)
    showblocks(
        backend,
        ("Sample" => getblocks(task), "Prediction" => predblock),
        collect(zip(samples, preds)),
    )
end

showpredictions(task::AbstractBlockMethod, args...) =
    showpredictions(default_showbackend(), task, args...)

"""
    showoutput([backend], task, output)
    showoutput([backend], task, encsample, output)

Show a model output to `backend`. If an encoded sample `encsample` is also
given, show it next to the output.
"""
function showoutput(backend::ShowBackend, task::AbstractBlockMethod, output)
    showblockinterpretable(
        backend,
        getencodings(task),
        "Output" => getblocks(task).ŷ,
        output,
    )
end
function showoutput(backend::ShowBackend, task::AbstractBlockMethod, encsample, output)
    blocks = getblocks(task)
    showblockinterpretable(
        backend,
        getencodings(task),
        ("Encoded sample" => blocks.encodedsample, "Output" => blocks.ŷ),
        (encsample, output),
    )
end
showoutput(task::AbstractBlockMethod, args...) =
    showoutput(default_showbackend(), task, args...)

"""
    showoutputs([backend], task, outputs)
    showoutputs([backend], task, encsamples, outputs)

Show model outputs to `backend`. If a vector of encoded samples `encsamples` is also
given, show them next to the outputs. Use [`showoutputbatch`](#) to show collated
batches of outputs.
"""
function showoutputs(backend::ShowBackend, task::AbstractBlockMethod, outputs)
    showblocksinterpretable(
        backend,
        getencodings(task),
        "Output" => getblocks(task).ŷ,
        outputs,
    )
end
function showoutputs(backend::ShowBackend, task::AbstractBlockMethod, encsamples, outputs)
    blocks = getblocks(task)
    showblocksinterpretable(
        backend,
        getencodings(task),
        ("Encoded sample" => blocks.encodedsample, "Output" => blocks.ŷ),
        collect(zip(encsamples, outputs)),
    )
end

showoutputs(task::AbstractBlockMethod, args...) =
    showoutputs(default_showbackend(), task, args...)


"""
    showoutputbatch([backend], task, outputbatch)
    showoutputbatch([backend], task, batch, outputbatch)

Show collated batch of outputs to `backend`. If a collated batch of encoded samples
`batch` is also given, show them next to the outputs. See [`showoutputs`](#) if you
have vectors of outputs and not collated batches.
"""
function showoutputbatch(backend::ShowBackend, task::AbstractBlockMethod, outputbatch)
    outputs = collect(DataLoaders.obsslices(outputbatch))
    return showoutputs(backend, task, outputs)
end
function showoutputbatch(
    backend::ShowBackend,
    task::AbstractBlockMethod,
    batch,
    outputbatch,
)
    encsamples = collect(DataLoaders.obsslices(batch))
    outputs = collect(DataLoaders.obsslices(outputbatch))
    return showoutputs(backend, task, encsamples, outputs)
end

showoutputbatch(task::AbstractBlockMethod, args...) =
    showoutputbatch(default_showbackend(), task, args...)

# Testing helper

"""
    test_task_show(task, backend::ShowBackend)

Test suite that tests that all learning task-related `show*` functions
work for `backend`

## Keyword arguments

- `sample = mockblock(getblocks(task))`: Sample data to use for tests.
- `output = mockblock(getblocks(task).ŷ)`: Model output data to use for tests.
"""
function test_task_show(
    task::LearningTask,
    backend::ShowBackend;
    sample = mockblock(getblocks(task).sample),
    output = mockblock(getblocks(task).ŷ),
    context = Training(),
)

    encsample = encodesample(task, context, sample)
    pred = decodeŷ(task, context, output)

    Test.@testset "`show*` test suite for learning task $task and backend $(typeof(backend))" begin
        @test (showsample(backend, task, sample); true)
        @test (showencodedsample(backend, task, encsample); true)
        @test (showoutput(backend, task, output); true)
        @test (showoutput(backend, task, encsample, output); true)
        @test (showprediction(backend, task, pred); true)
        @test (showprediction(backend, task, sample, pred); true)
    end
end

# Deprecations
