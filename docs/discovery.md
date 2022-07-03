# Discovery

As you may have seen in [the introduction](./introduction.md), FastAI.jl makes it possible to train models in just 5 lines of code. However, if you have a task in mind, you need to know what datasets you can train on and if there are convenience learning task constructors. For example, the introduction loads the `"imagenette2-160"` dataset and uses [`ImageClassificationSingle`](#) to construct a learning task. Now what if, instead of classifying an image into one class, we want to classify every single pixel into a class (semantic segmentation)? Now we need a dataset with pixel-level annotations and a learning task that can process those segmentation masks.

For finding both, we can make use of `Block`s. A `Block` represents a kind of data, for example images, labels or keypoints. For supervised learning tasks, we have an input block and a target block. If we wanted to classify whether 2D images contain a cat or a dog, we could use the blocks `(Image{2}(), Label(["cat", "dog"]))`, while for semantic segmentation, we'll have an input `Image` block and a target [`Mask`](#) block.

## Finding a dataset

To find a dataset with compatible samples, we can pass the types of these blocks as a filter to [`datasets`](#) which will show us only dataset recipes for loading those blocks.

{cell=main}
```julia
using FastAI, FastVision
datarecipes(blocks=(Image, Mask))
```

We can see that the `"camvid_tiny"` dataset can be loaded so that each sample is a pair of an image and a segmentation mask. Let's use a data recipe to load a [data container](data_containers.md) and concrete blocks.

{cell=main, result=false, output=false style="display:none;"}
```julia
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
```

{cell=main, output=false}
```julia
data, blocks = load(findfirst(datarecipes(id="camvid_tiny", blocks=(Image, Mask))))
```

As with every data container, we can load a sample using `getobs` which gives us a tuple of an image and a segmentation mask.

{cell=main}
```julia
image, mask = sample = getobs(data, 1)
size.(sample), eltype.(sample)
```

Loading the dataset recipe also returned `blocks`, which are the concrete [`Block`] instances for the dataset. We passed in _types_ of blocks (`(Image, Mask)`) and get back _instances_ since the specifics of some blocks depend on the dataset. For example, the returned target block carries the labels for every class that a pixel can belong to.

{cell=main}
```julia
inputblock, targetblock = blocks
targetblock
```

With these `blocks`, we can also validate a sample of data using [`checkblock`](#) which is useful as a sanity check when using custom data containers.

{cell=main}
```julia
checkblock((inputblock, targetblock), (image, mask))
```

### Summary

In short, if you have a learning task in mind and want to load a dataset for that task, then

1. define the types of input and target block, e.g. `blocktypes = (Image, Label)`,
2. use `filter(`[`datarecipes`](#)`(), blocks=blocktypes)` to find compatbile dataset recipes; and
3. run `load(`[`datarecipes`](#)`()[id])` to load a data container and the concrete blocks

### Exercises

1. Find and load a dataset for multi-label image classification. (Hint: the block for multi-category outputs is called `LabelMulti`).
2. List all datasets with `Image` as input block and any target block. (Hint: the supertype of all types is `Any`)


## Finding a learning task

Armed with a dataset, we can go to the next step: creating a learning task. Since we already have blocks defined, this amounts to defining the encodings that are applied to the data before it is used in training. Here, FastAI.jl already defines some convenient constructors for learning tasks and you can find them with [`learningtasks`](#). Here we can pass in either block types as above or the block instances:

{cell=main}
```julia
learningtasks(blocks=blocks)
```

Looks like we can use the [`ImageSegmentation`](#) function to create a learning task. Every function returned can be called with `blocks` and, optionally, some keyword arguments for customization.

{cell=main}
```julia
task = ImageSegmentation(blocks; size = (64, 64))
```

And that's the basic workflow for getting started with a supervised task.

### Exercises

1. Find all learning task functions with images as inputs.
