# High-level plotting functions for use with `BlockMethod`s

"""
    showsample([backend], method, sample)

Show an unprocessed `sample` for `LearningMethod` `method` to
`backend::`[`ShowBackend`](#).

## Examples

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
method = ImageClassificationSingle(data)
sample = getobs(data, 1)
showsample(method, sample)  # select backend automatically
showsample(ShowText(), method, sample)
```
"""
function showsample(backend::ShowBackend, method::BlockMethod, sample)
    blocks = ("Input" => method.blocks[1], "Target" => method.blocks[2])
    showblock(backend, blocks, sample)
end
showsample(method::BlockMethod, sample) = showsample(default_showbackend(), method, sample)


"""
    showsample([backend], method, sample)

Show a vector of unprocessed `samples` for `LearningMethod` `method` to
`backend::`[`ShowBackend`](#).

## Examples

```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
method = ImageClassificationSingle(data)
samples = [getobs(data, i) for i in 1:4]
showsamples(method, samples)  # select backend automatically
showsamples(ShowText(), method, samples)
```
"""
function showsamples(backend::ShowBackend, method::BlockMethod, samples)
    blocks = ("Input" => method.blocks[1], "Target" => method.blocks[2])
    showblocks(backend, blocks, samples)
end
showsamples(method::BlockMethod, samples) = showsamples(default_showbackend(), method, sample)

"""
    showencodedsample([backend], method, encsample)

Show an encoded sample `encsample` to `backend`.
"""
function showencodedsample(backend::ShowBackend, method::BlockMethod, encsample)
    xblock, yblock = encodedblock(method.encodings, method.blocks, true)
    showblockinterpretable(
        backend,
        method.encodings,
        ("x" => xblock, "y" => yblock),
        encsample)
end
showencodedsample(method, encsample) = showencodedsample(default_showbackend(), method, encsample)

"""
    showencodedsamples([backend], method, encsamples)

Show a vector of encoded samples `encsamples` to `backend`.
"""
function showencodedsamples(backend::ShowBackend, method::BlockMethod, encsamples::AbstractVector)
    xblock, yblock = encodedblock(method.encodings, method.blocks, true)
    showblocksinterpretable(
        backend,
        method.encodings,
        ("x" => xblock, "y" => yblock),
        encsamples)
end

"""
    showbatch([backend], method, batch)

Show a collated batch of encoded samples to `backend`.
"""
function showbatch(backend::ShowBackend, method::BlockMethod, batch)
    encsamples = collect(DataLoaders.obsslices(batch))
    showencodedsamples(backend, method, encsamples)
end
showbatch(method, batch) = showbatch(default_showbackend(), method, batch)

"""
    showprediction([backend], method, pred)
    showprediction([backend], method, sample, pred)

Show a prediction `pred`. If a `sample` is also given, show it next to
the prediction.
"""
function showprediction(backend::ShowBackend, method::BlockMethod, pred)
    predblock = decodedblock(method.encodings, method.outputblock, true)
    showblock(backend, "Prediction" => predblock, pred)
end

function showprediction(backend::ShowBackend, method::BlockMethod, sample, pred)
    predblock = decodedblock(method.encodings, method.outputblock, true)
    showblock(
        backend,
        ("Sample" => method.blocks, "Prediction" => predblock),
        (sample, pred)
    )
end


showprediction(method::BlockMethod, args...) =
    showprediction(default_showbackend(), method, args...)

"""
    showpredictions([backend], method, preds)
    showpredictions([backend], method, samples, preds)

Show predictions `pred`. If `samples` are also given, show them next to
the prediction.
"""
function showpredictions(backend::ShowBackend, method::BlockMethod, preds)
    predblock = decodedblock(method.encodings, method.outputblock, true)
    showblocks(backend, "Prediction" => predblock, preds)
end

function showpredictions(backend::ShowBackend, method::BlockMethod, samples, preds)
    predblock = decodedblock(method.encodings, method.outputblock, true)
    showblocks(
        backend,
        ("Sample" => method.blocks, "Prediction" => predblock),
        collect(zip(samples, preds)),
    )
end

showpredictions(method::BlockMethod, args...) =
    showpredictions(default_showbackend(), method, args...)

"""
    showoutput([backend], method, output)
    showoutput([backend], method, encsample, output)

Show a model output to `backend`. If an encoded sample `encsample` is also
given, show it next to the output.
"""
function showoutput(backend::ShowBackend, method::BlockMethod, output)
    showblockinterpretable(backend, method.encodings, "Output" => method.outputblock, output)
end
function showoutput(backend::ShowBackend, method::BlockMethod, encsample, output)
    encsampleblock = encodedblock(method.encodings, method.blocks, true)
    outblock = method.outputblock
    showblockinterpretable(
        backend,
        method.encodings,
        ("Encoded sample" => encsampleblock, "Output" => outblock),
        (encsample, output))
end
showoutput(method::BlockMethod, args...) = showoutput(default_showbackend(), method, args...)

"""
    showoutputs([backend], method, outputs)
    showoutputs([backend], method, encsamples, outputs)

Show model outputs to `backend`. If a vector of encoded samples `encsamples` is also
given, show them next to the outputs. Use [`showoutputbatch`](#) to show collated
batches of outputs.
"""
function showoutputs(backend::ShowBackend, method::BlockMethod, outputs)
    showblocks(backend, "Output" => method.outputblock, outputs)
end
function showoutputs(backend::ShowBackend, method::BlockMethod, encsamples, outputs)
    encsampleblock = encodedblock(method.encodings, method.blocks, true)
    outblock = method.outputblock
    showblocksinterpretable(
        backend,
        method.encodings,
        ("Encoded sample" => encsampleblock, "Output" => outblock),
        (encsamples, outputs))
end

showoutputs(method::BlockMethod, args...) = showoutputs(default_showbackend(), method, args...)


"""
    showoutputbatch([backend], method, outputbatch)
    showoutputbatch([backend], method, batch, outputbatch)

Show collated batch of outputs to `backend`. If a collated batch of encoded samples
`batch` is also given, show them next to the outputs. See [`showoutputs`](#) if you
have vectors of outputs and not collated batches.
"""
function showoutputbatch(backend::ShowBackend, method::BlockMethod, outputbatch)
    outputs = collect(DataLoaders.obsslices(outputbatch))
    return showoutputs(backend, method, outputs)
end
function showoutputbatch(backend::ShowBackend, method::BlockMethod, batch, outputbatch)
    encsamples = collect(DataLoaders.obsslices(batch))
    outputs = collect(DataLoaders.obsslices(outputbatch))
    return showoutputs(backend, method, encsamples, outputs)
end

showoutputbatch(method::BlockMethod, args...) = showoutputbatch(default_showbackend(), method, args...)

# Testing helper

"""
    test_method_show(method, backend::ShowBackend)

Test suite that tests that all learning method-related `show*` functions
work for `backend`

## Keyword arguments

- `sample = mockblock(method.blocks)`: Sample data to use for tests.
- `output = mockblock(method.outputblock)`: Model output data to use for tests.
"""
function test_method_show(
        method::LearningMethod, backend::ShowBackend;
        sample = mockblock(method.blocks),
        output = mockblock(method.outputblock),
        context = Training())

    encsample = encode(method, context, sample)
    pred = decodeŷ(method, context, output)

    Test.@testset "`show*` test suite for learning method $method and backend $(typeof(backend))" begin
        Test.@test_nowarn showsample(backend, method, sample)
        Test.@test_nowarn showencodedsample(backend, method, encsample)
        Test.@test_nowarn showoutput(backend, method, output)
        Test.@test_nowarn showoutput(backend, method, encsample, output)
        Test.@test_nowarn showprediction(backend, method, pred)
        Test.@test_nowarn showprediction(backend, method, sample, pred)
    end
end

# Deprecations
