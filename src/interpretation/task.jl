# High-level plotting functions for use with `BlockTask`s

"""
    showsample([backend], task, sample)

Show an unprocessed `sample` for `LearningTask` `task` to
`backend::`[`ShowBackend`](#).

## Examples

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
task = ImageClassificationSingle(data)
sample = data[1]
showsample(task, sample)  # select backend automatically
showsample(ShowText(), task, sample)
```
"""
function showsample(backend::ShowBackend, task::AbstractBlockTask, sample)
    blocks = ("Input" => getblocks(task)[1], "Target" => getblocks(task)[2])
    showblock(backend, blocks, sample)
end
function showsample(task::AbstractBlockTask, sample)
    showsample(default_showbackend(), task, sample)
end

"""
    showsample([backend], task, sample)

Show a vector of unprocessed `samples` for `LearningTask` `task` to
`backend::`[`ShowBackend`](#).

## Examples

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
task = ImageClassificationSingle(data)
samples = [data[i] for i in 1:4]
showsamples(task, samples)  # select backend automatically
showsamples(ShowText(), task, samples)
```
"""
function showsamples(backend::ShowBackend, task::AbstractBlockTask, samples)
    showblocks(backend, "Sample" => getblocks(task).sample, samples)
end
function showsamples(task::AbstractBlockTask, samples)
    showsamples(default_showbackend(), task, sample)
end

"""
    showencodedsample([backend], task, encsample)

Show an encoded sample `encsample` to `backend`.
"""
function showencodedsample(backend::ShowBackend, task::AbstractBlockTask, encsample)
    showblockinterpretable(backend,
                           getencodings(task),
                           getblocks(task).encodedsample,
                           encsample)
end
function showencodedsample(task, encsample)
    showencodedsample(default_showbackend(), task, encsample)
end

"""
    showencodedsamples([backend], task, encsamples)

Show a vector of encoded samples `encsamples` to `backend`.
"""
function showencodedsamples(backend::ShowBackend,
                            task::AbstractBlockTask,
                            encsamples::AbstractVector)
    xblock, yblock = encodedblockfilled(getencodings(task), getblocks(task).encodedsample)
    showblocksinterpretable(backend,
                            getencodings(task),
                            ("x" => xblock, "y" => yblock),
                            encsamples)
end

"""
    showbatch([backend], task, batch)

Show a collated batch of encoded samples to `backend`.
"""
function showbatch(backend::ShowBackend, task::AbstractBlockTask, batch)
    showencodedsamples(backend, task, Datasets.unbatch(batch))
end
showbatch(task, batch) = showbatch(default_showbackend(), task, batch)

"""
    showprediction([backend], task, pred)
    showprediction([backend], task, sample, pred)

Show a prediction `pred`. If a `sample` is also given, show it next to
the prediction. ŷ
"""
function showprediction(backend::ShowBackend, task::AbstractBlockTask, pred)
    showblock(backend, "Prediction" => getblocks(task).pred, pred)
end

function showprediction(backend::ShowBackend, task::AbstractBlockTask, sample, pred)
    blocks = getblocks(task)
    showblock(backend,
              ("Sample" => blocks.sample, "Prediction" => blocks.pred),
              (sample, pred))
end

function showprediction(task::AbstractBlockTask, args...)
    showprediction(default_showbackend(), task, args...)
end

"""
    showpredictions([backend], task, preds)
    showpredictions([backend], task, samples, preds)

Show predictions `pred`. If `samples` are also given, show them next to
the prediction.
"""
function showpredictions(backend::ShowBackend, task::AbstractBlockTask, preds)
    predblock = decodedblockfilled(getencodings(task), getblocks(task).ŷ)
    showblocks(backend, "Prediction" => predblock, preds)
end

function showpredictions(backend::ShowBackend, task::AbstractBlockTask, samples, preds)
    predblock = decodedblockfilled(getencodings(task), getblocks(task).ŷ)
    showblocks(backend,
               ("Sample" => getblocks(task), "Prediction" => predblock),
               collect(zip(samples, preds)))
end

function showpredictions(task::AbstractBlockTask, args...)
    showpredictions(default_showbackend(), task, args...)
end

"""
    showoutput([backend], task, output)
    showoutput([backend], task, encsample, output)

Show a model output to `backend`. If an encoded sample `encsample` is also
given, show it next to the output.
"""
function showoutput(backend::ShowBackend, task::AbstractBlockTask, output)
    showblockinterpretable(backend,
                           getencodings(task),
                           "Output" => getblocks(task).ŷ,
                           output)
end
function showoutput(backend::ShowBackend, task::AbstractBlockTask, encsample, output)
    blocks = getblocks(task)
    showblockinterpretable(backend,
                           getencodings(task),
                           ("Encoded sample" => blocks.encodedsample,
                            "Output" => blocks.ŷ),
                           (encsample, output))
end
function showoutput(task::AbstractBlockTask, args...)
    showoutput(default_showbackend(), task, args...)
end

"""
    showoutputs([backend], task, outputs)
    showoutputs([backend], task, encsamples, outputs)

Show model outputs to `backend`. If a vector of encoded samples `encsamples` is also
given, show them next to the outputs. Use [`showoutputbatch`](#) to show collated
batches of outputs.
"""
function showoutputs(backend::ShowBackend, task::AbstractBlockTask, outputs)
    showblocksinterpretable(backend,
                            getencodings(task),
                            "Output" => getblocks(task).ŷ,
                            outputs)
end
function showoutputs(backend::ShowBackend, task::AbstractBlockTask, encsamples, outputs)
    blocks = getblocks(task)
    showblocksinterpretable(backend,
                            getencodings(task),
                            ("Encoded sample" => blocks.encodedsample,
                             "Output" => blocks.ŷ),
                            collect(zip(encsamples, outputs)))
end

function showoutputs(task::AbstractBlockTask, args...)
    showoutputs(default_showbackend(), task, args...)
end

"""
    showoutputbatch([backend], task, outputbatch)
    showoutputbatch([backend], task, batch, outputbatch)

Show collated batch of outputs to `backend`. If a collated batch of encoded samples
`batch` is also given, show them next to the outputs. See [`showoutputs`](#) if you
have vectors of outputs and not collated batches.
"""
function showoutputbatch(backend::ShowBackend, task::AbstractBlockTask, outputbatch)
    return showoutputs(backend, task, Datasets.unbatch(outputbatch))
end
function showoutputbatch(backend::ShowBackend,
                         task::AbstractBlockTask,
                         batch,
                         outputbatch)
    return showoutputs(backend, task, Datasets.unbatch(batch),
                       Datasets.unbatch(outputbatch))
end

function showoutputbatch(task::AbstractBlockTask, args...)
    showoutputbatch(default_showbackend(), task, args...)
end

# Testing helper

"""
    test_task_show(task, backend::ShowBackend)

Test suite that tests that all learning task-related `show*` functions
work for `backend`

## Keyword arguments

- `sample = mockblock(getblocks(task))`: Sample data to use for tests.
- `output = mockblock(getblocks(task).ŷ)`: Model output data to use for tests.
"""
function test_task_show(task::LearningTask,
                        backend::ShowBackend;
                        sample = mockblock(getblocks(task).sample),
                        output = mockblock(getblocks(task).ŷ),
                        context = Training())
    encsample = encodesample(task, context, sample)
    pred = decodeypred(task, context, output)

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
