# Blocks and encodings

_Unstructured notes on blocks and encodings_

## Blocks

> A **block** describes the meaning of a piece of data in the context of a learning task.

- For example, for supervised learning tasks, there is an input block and a target block and we want to learn to predict targets from inputs. Learning to predict a cat/dog label (`Label(["cat", "dog"])`) from 2D images (`Image{2}()`) is a supervised image classification task.
- A block is not a piece of data itself. Instead it describes the meaning of a piece of data in a context. That a piece of data is a block can be checked using [`checkblock`]`(block, data)`. A piece of data for the `Label` block above needs to be one of the labels, so `checkblock(Label(["cat", "dog"]), "cat") == true`, but `checkblock(Label(["cat", "dog"]), "cat") == false`.
- We can say that a data container is compatible with a learning task if every observation in it is a valid sample of the sample block of the learning task. The sample block for supervised tasks is `sampleblock = (inputblock, targetblock)` so `sample = getobs(data, i)` from a compatible data container implies that `checkblock(sampleblock, sample)`. This also means that any data stored in blocks must not depend on individual samples; we can store the names of possible classes inside the `Label` block because they are the same across the whole dataset.


## Data pipelines

We can use blocks to formalize the data processing pipeline.

During **training** we want to create pairs of data `(x, y)` s.t. `output = model(x)` and `loss = lossfn(output, y)`. In terms of blocks that means `model` is a function `(x,) -> output` and the loss function maps `(outputblock, yblock) -> loss`. Usually, `(input, target) != (x, y)` and instead we have an encoding step that transforms a sample into representations suitable to train a model on, i.e. `encode :: sample -> (x, y)`.

- For the above image classification example we have `sampleblock = (Image{2}(), Label(["cat", "dog"]))` but we cannot put raw images into a model and get out a class. Instead, the image is converted to an array that includes the color dimension and its values are normalized; and the class label is one-hot encoded. So `xblock = ImageTensor{2}()` and `yblock = OneHotTensor{0}`. Hence to do training, we need a sample encoding function `(Image{2}, Label) -> (ImageTensor{2}, OneHotTensor{0})`

During **inference**, we have an input and want to use a trained model to predict a target, i.e. `input -> target`. The model is again a mapping `xblock -> outputblock`, so we can build the transformation with an encoding step that encodes the input and a decoding step that takes the model output back into a target. 

This gives us
> `(predict :: input -> target) = decodeoutput ∘ model ∘ encodeinput`
> where 
> - `(encodeinput :: input -> x)`
> - `(model :: x -> y)`
> - `(decodeoutput :: y -> target)`

- In the classification example we have, written in blocks, `predict :: Image{2} -> Label` and hence `encodeinput :: Image{2} -> ImageTensor{2}` and `decodeoutput :: OneHotTensor{0} -> Label`

Where do we draw the line between model and data processing? In general, the encoding and decoding steps are **non-learnable** transformations, while the model is a **learnable** transformation.

## Encodings

> **Encodings** are reversible transformations that model the non-learnable parts (encoding and decoding) of the data pipeline.

- What an encoding does depends on what block is passed in. Most encodings only transform specific blocks. For example, the [`ImagePreprocessing`](#) encoding maps blocks `Image{N} -> ImageTensor{N}`, but leaves other blocks unchanged. Encodings are called with `encode` and `decode` which take in the block and the data. The actual encoding and decoding takes in an additional context argument which can be specialized on to implement different behavior for e.g. training and validation.
    {cell=main}
    ```julia
    using FastAI, Colors
    using FastAI.Vision: ImageTensor
    enc = ImagePreprocessing()
    data = rand(RGB, 100, 100)
    @show summary(data)
    encdata = encode(enc, Training(), Image{2}(), data)
    @show summary(encdata)  # (h, w, ch)-image tensor
    data_ = decode(enc, Training(), ImageTensor{2}(3), encdata)
    ```
- Using an encoding to encode and then decode must be block-preserving, i.e. if, for an encoding, `encode :: Block1 -> Block2` then `decode :: Block2 -> Block1`. To see the resulting block of applying an encoding to a block, we can use [`encodedblock`](#) and [`decodedblock`](#).
    {cell=main}
    ```julia
    using FastAI: encodedblock, decodedblock
    enc = ImagePreprocessing()
    @show encodedblock(enc, Image{2}())
    @show decodedblock(enc, ImageTensor{2}(3))
    Image{2}() == decodedblock(enc, encodedblock(enc, Image{2}()))
    ```
    You can use [`testencoding`](#) to test these invariants to make sure an encoding is implemented properly for a specific block.
    {cell=main}

    ```julia
    FastAI.testencoding(enc, Image{2}())
    ```
- The default implementations of `encodedblock` and `decodedblock` is to return `nothing` indicating that it doesn't transform the data. This is overwritten for blocks for which `encode` and `decode` are implemented to indicate that the data is transformed. Using `encodedblockfilled(block, data)` will replace returned `nothing`s with the unchanged block.
    {cell=main}
    ```julia
    encodedblock(enc, Label(1:10)) === nothing
    ```
    {cell=main}
    ```julia
    using FastAI: encodedblockfilled
    encodedblockfilled(enc, Label(1:10)) == Label(1:10)
    ```
- Encodings can be applied to tuples of blocks. The default behavior is to apply the encoding to each block separately.
    {cell=main}
    ```julia
    encodedblock(enc, (Image{2}(), Image{2}()))
    ```

- Applying a tuple of encodings will encode the data by applying one encoding after the other. When decoding, the order is reversed.

## Block learning tasks

[`BlockTask`](#) creates a learning task from blocks and encodings. You define the sample block (recall for supervised tasks this is a tuple of input and target) and a sequence of encodings that are applied to all blocks.

The below example defines the same learning task as [`ImageClassificationSingle`](#) does. The first two encodings only change `Image`, and the last changes only `Label`, so it's simple to understand.

{cell=main}
```julia
task = BlockTask(
    (Image{2}(), Label(["cats", "dogs"])),
    (
        ProjectiveTransforms((128, 128)),
        ImagePreprocessing(),
        OneHot(),
    )
)
```

Now `encode` expects a sample and just runs the encodings over that, giving us an encoded input `x` and an encoded target `y`.

```julia
data = loadfolderdata(joinpath(load(datasets()["dogscats"]), "train"), filterfn=isimagefile, loadfn=(loadfile, parentname))
sample = getobs(data, 1)
x, y = encodesample(task, Training(), sample)
summary(x), summary(y)
```

This is equivalent to:

```julia
x, y = encode(task.encodings, Training(), FastAI.getblocks(task).sample, sample)
summary(x), summary(y)
```

Image segmentation looks almost the same except we use a `Mask` block as target. We're also using `OneHot` here, because it also has an `encode` task for `Mask`s. For this task, `ProjectiveTransforms` will be applied to both the `Image` and the `Mask`, using the same random state for cropping and augmentation.

```julia
task = BlockTask(
    (Image{2}(), Mask{2}(1:10)),
    (
        ProjectiveTransforms((128, 128)),
        ImagePreprocessing(),
        OneHot(),
    )
)
```

The easiest way to understand how encodings are applied to each block is to use [`describetask`](#) and [`describeencodings`](#) which print a table of how each encoding is applied successively to each block. Rows where a block is **bolded** indicate that the data was transformed by that encoding.

```julia
describetask(task)
```

The above tables make it clear what happens during training ("encoding a sample") and inference (encoding an input and "decoding an output"). The more general form [`describeencodings`](#) takes in encodings and blocks directly and can be useful for building an understanding of how encodings apply to some blocks.

```julia
FastAI.describeencodings(task.encodings, (Image{2}(),))
```

```julia
FastAI.describeencodings((OneHot(),), (Label(1:10), Mask{2}(1:10), Image{2}()))
```

Notes

- Since most encodings just operate on a small number of blocks and keep the rest unchanged, applying them to all blocks is usually not a problem. When it is because you want some encoding to apply to a specific block only, you can use [`Named`](#) and [`Only`](#) to get around it.
