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
function showsample(backend::ShowBackend, method::AbstractBlockMethod, sample)
    blocks = ("Input" => getblocks(method)[1], "Target" => getblocks(method)[2])
    showblock(backend, blocks, sample)
end
showsample(method::AbstractBlockMethod, sample) =
    showsample(default_showbackend(), method, sample)


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
function showsamples(backend::ShowBackend, method::AbstractBlockMethod, samples)
    showblocks(backend, "Sample" => getblocks(method).sample, samples)
end
showsamples(method::AbstractBlockMethod, samples) =
    showsamples(default_showbackend(), method, sample)

"""
    showencodedsample([backend], method, encsample)

Show an encoded sample `encsample` to `backend`.
"""
function showencodedsample(backend::ShowBackend, method::AbstractBlockMethod, encsample)
    showblockinterpretable(
        backend,
        getencodings(method),
        "Encoded sample" => getblocks(method).encodedsample,
        encsample,
    )
end
showencodedsample(method, encsample) =
    showencodedsample(default_showbackend(), method, encsample)

"""
    showencodedsamples([backend], method, encsamples)

Show a vector of encoded samples `encsamples` to `backend`.
"""
function showencodedsamples(
    backend::ShowBackend,
    method::AbstractBlockMethod,
    encsamples::AbstractVector,
)
    xblock, yblock = encodedblockfilled(getencodings(method), getblocks(method))
    showblocksinterpretable(
        backend,
        getencodings(method),
        ("x" => xblock, "y" => yblock),
        encsamples,
    )
end

"""
    showbatch([backend], method, batch)

Show a collated batch of encoded samples to `backend`.
"""
function showbatch(backend::ShowBackend, method::AbstractBlockMethod, batch)
    encsamples = collect(DataLoaders.obsslices(batch))
    showencodedsamples(backend, method, encsamples)
end
showbatch(method, batch) = showbatch(default_showbackend(), method, batch)

"""
    showprediction([backend], method, pred)
    showprediction([backend], method, sample, pred)

Show a prediction `pred`. If a `sample` is also given, show it next to
the prediction. ŷ
"""
function showprediction(backend::ShowBackend, method::AbstractBlockMethod, pred)
    showblock(backend, "Prediction" => getblocks(method).pred, pred)
end

function showprediction(backend::ShowBackend, method::AbstractBlockMethod, sample, pred)
    blocks = getblocks(method)
    showblock(
        backend,
        ("Sample" => blocks.sample, "Prediction" => blocks.pred),
        (sample, pred),
    )
end


showprediction(method::AbstractBlockMethod, args...) =
    showprediction(default_showbackend(), method, args...)

"""
    showpredictions([backend], method, preds)
    showpredictions([backend], method, samples, preds)

Show predictions `pred`. If `samples` are also given, show them next to
the prediction.
"""
function showpredictions(backend::ShowBackend, method::AbstractBlockMethod, preds)
    predblock = decodedblockfilled(getencodings(method), getblocks(method).ŷ)
    showblocks(backend, "Prediction" => predblock, preds)
end

function showpredictions(backend::ShowBackend, method::AbstractBlockMethod, samples, preds)
    predblock = decodedblockfilled(getencodings(method), getblocks(method).ŷ)
    showblocks(
        backend,
        ("Sample" => getblocks(method), "Prediction" => predblock),
        collect(zip(samples, preds)),
    )
end

showpredictions(method::AbstractBlockMethod, args...) =
    showpredictions(default_showbackend(), method, args...)

"""
    showoutput([backend], method, output)
    showoutput([backend], method, encsample, output)

Show a model output to `backend`. If an encoded sample `encsample` is also
given, show it next to the output.
"""
function showoutput(backend::ShowBackend, method::AbstractBlockMethod, output)
    showblockinterpretable(
        backend,
        getencodings(method),
        "Output" => getblocks(method).ŷ,
        output,
    )
end
function showoutput(backend::ShowBackend, method::AbstractBlockMethod, encsample, output)
    blocks = getblocks(method)
    showblockinterpretable(
        backend,
        getencodings(method),
        ("Encoded sample" => blocks.encodedsample, "Output" => blocks.ŷ),
        (encsample, output),
    )
end
showoutput(method::AbstractBlockMethod, args...) =
    showoutput(default_showbackend(), method, args...)

"""
    showoutputs([backend], method, outputs)
    showoutputs([backend], method, encsamples, outputs)

Show model outputs to `backend`. If a vector of encoded samples `encsamples` is also
given, show them next to the outputs. Use [`showoutputbatch`](#) to show collated
batches of outputs.
"""
function showoutputs(backend::ShowBackend, method::AbstractBlockMethod, outputs)
    showblocksinterpretable(
        backend,
        getencodings(method),
        "Output" => getblocks(method).ŷ,
        outputs,
    )
end
function showoutputs(backend::ShowBackend, method::AbstractBlockMethod, encsamples, outputs)
    blocks = getblocks(method)
    showblocksinterpretable(
        backend,
        getencodings(method),
        ("Encoded sample" => blocks.encodedsample, "Output" => blocks.ŷ),
        collect(zip(encsamples, outputs)),
    )
end

showoutputs(method::AbstractBlockMethod, args...) =
    showoutputs(default_showbackend(), method, args...)


"""
    showoutputbatch([backend], method, outputbatch)
    showoutputbatch([backend], method, batch, outputbatch)

Show collated batch of outputs to `backend`. If a collated batch of encoded samples
`batch` is also given, show them next to the outputs. See [`showoutputs`](#) if you
have vectors of outputs and not collated batches.
"""
function showoutputbatch(backend::ShowBackend, method::AbstractBlockMethod, outputbatch)
    outputs = collect(DataLoaders.obsslices(outputbatch))
    return showoutputs(backend, method, outputs)
end
function showoutputbatch(
    backend::ShowBackend,
    method::AbstractBlockMethod,
    batch,
    outputbatch,
)
    encsamples = collect(DataLoaders.obsslices(batch))
    outputs = collect(DataLoaders.obsslices(outputbatch))
    return showoutputs(backend, method, encsamples, outputs)
end

showoutputbatch(method::AbstractBlockMethod, args...) =
    showoutputbatch(default_showbackend(), method, args...)

# Testing helper

"""
    test_method_show(method, backend::ShowBackend)

Test suite that tests that all learning method-related `show*` functions
work for `backend`

## Keyword arguments

- `sample = mockblock(getblocks(method))`: Sample data to use for tests.
- `output = mockblock(getblocks(method).ŷ)`: Model output data to use for tests.
"""
function test_method_show(
    method::LearningMethod,
    backend::ShowBackend;
    sample = mockblock(getblocks(method).sample),
    output = mockblock(getblocks(method).ŷ),
    context = Training(),
)

    encsample = encode(method, context, sample)
    pred = decodeŷ(method, context, output)

    Test.@testset "`show*` test suite for learning method $method and backend $(typeof(backend))" begin
        @test (showsample(backend, method, sample); true)
        @test (showencodedsample(backend, method, encsample); true)
        @test (showoutput(backend, method, output); true)
        @test (showoutput(backend, method, encsample, output); true)
        @test (showprediction(backend, method, pred); true)
        @test (showprediction(backend, method, sample, pred); true)
    end
end

# Deprecations
