# fastai API comparison 

FastAI.jl is in many ways similar to the original Python [fastai](http://docs.fast.ai), but also has its differences. This reference goes through all the sections in the [fastai: A Layered API for Deep Learning](https://arxiv.org/abs/2002.04688) paper and comments what the interfaces for the same functionality in FastAI.jl are, and where they differ or functionality is still missing.

## Applications

FastAI.jl's own data block API makes it possible to derive every part of a high-level interface with a unified API across tasks. Instead it suffices to create a learning task and based on the blocks and encodings specified the proper model builder, loss function, and visualizations are implemented (see below). For a high-level API, a complete `Learner` can be constructed using [`tasklearner`](#) without much boilerplate. There are some helper functions for  creating these learning tasks, for example [`ImageClassificationSingle`](#) and [`ImageSegmentation`](#).

FastAI.jl additionally has a unified API for registering and discovering functionality across applications also based on the data block abstraction.  [`datasets`](#) and [`datarecipes`](#) let you quickly load common datasets matching some data modality and [`learningtasks`] lets you find learning task helpers for common tasks. See [the discovery tutorial](discovery.md) for more info.

### Vision

Computer vision is well-supported in FastAI.jl with different tasks and optimized data pipelines for N-dimensional images, masks and keypoints. See the tutorial section for many examples.

### Tabular

FastAI.jl also has support for tabular data.

### Deployment

Through FastAI.jl's [`LearningTask`](#) interface, the data processing logic is decoupled from the dataset creation and training and can be easily serialized and loaded to make predictions. See the tutorial on [saving and loading models](../notebooks/serialization.ipynb).


---

There is no integration (yet!) for text and collaborative filtering applications.

## High-level API

### High-level API foundations

FastAI.jl also has a data block API but it differs from fastai's in a number of ways. In the Julia package it only handles the data encoding and decoding part, and doesn't concern itself with creating datasets. For dataset loading, see the [data container API](data_containers.md). As mentioned above, the high-level application-specific logic is also derived from the data block API. To use it you need to specify a tuple of input and target blocks as well as a tuple of encodings that are applied to the data. The encodings  are invertible data-specific data processing steps which correspond to `fastai.Transform`s. As in fastai, dispatch is used to transform applicable data and pass other data through unchanged. Unlike in fastai, there are no default steps associated with a block, allowing greater flexibility.

We can create a `BlockTask` (similar to `fastai.DataBlock`) and get information about the representations the data goes through.

{cell=main}
```julia
using FastAI, FastVision

task = BlockTask(
    (Image{2}(), Mask{2}(["foreground", "background"])),
    (
        ProjectiveTransforms((128, 128)),
        ImagePreprocessing(),
        OneHot(),
    )
)
describetask(task)
```

From this short definition, many things can be derived:

- data encoding
- model output decoding
- how to create a model from a backbone
- the loss function to use
- how to visualize samples and predictions

Together with a [data container](data_container) `data`, we can quickly create a `Learner` using [`tasklearner`](#) which, like in fastai, handles the training for us. There are no application-specific `Learner` constructors like `cnn_learner` or `unet_learner` in FastAI.jl.

```julia
learner = tasklearner(task, data)
```

High-level training protocols like the [one-cycle learning rate schedule](../notebooks/fitonecycle.ipynb), [fine-tuning](../notebooks/finetune.ipynb) and the [learning rate finder](../notebooks/lrfind.ipynb) are then available to us:

```julia
fit!(learner, 10)                  # Basic training for 10 epochs
finetune!(learner, 5, 1e-3)        # Finetuning regimen for 1+5 epochs with lr=1e-3
fitonecycle!(learner, 10)          # One-cycle learning rate regimen
res = lrfind(learner); plot(res)   # Run learning rate finder and plot suggestions
```

### Incrementally adapting PyTorch code

Since it is a Julia package, FastAI.jl is not written on top of PyTorch, but a Julia library for deep learning: [Flux.jl](http://www.fluxml.ai). In any case, the point of this section is to note that the abstractions in fastai are decoupled and existing projects can easily be reused. This is also the case for FastAI.jl as it is built on top of several decoupled libraries. Many of these were built specifically for FastAI.jl, but they are unaware of each other and useful in their own right:

- [Flux.jl](https://github.com/FluxML/Flux.jl) provides models, optimizers, and loss functions, fulfilling a similar role to PyTorch
- [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) gives you tools for building and transforming data containers. Also, it takes care of efficient, parallelized iteration of data containers.
- [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl) takes care of the lower levels of high-performance, composable data augmentations.
- [FluxTraining.jl](https://github.com/lorenzoh/FluxTraining.jl) contributes a highly extensible training loop with 2-way callbacks

If that seems like a lot: don't worry! If you've installed FastAI.jl, the functionality of most of these packages is reexported and you don't have to install any of them explicitly.

### Consistency across domains

While computer vision is the only domain with mature support for now, the abstractions underlying FastAI.jl are carefully crafted to ensure that learning tasks for different domains can be created using the same set of interfaces. This shows in that there's no need for application-specific functionality above the data block API.

## Mid-level APIs

### Learner

The [`Learner`](#) is very similar to fastai's. It takes

- a model: any parameterized, differentiable function like a neural network or even [a trebuchet simulator](https://fluxml.ai/blog/2019/03/05/dp-vs-rl.html)
- training and validation data iterators: these can be `DataLoader`s which paralellize data loading but any iterator over batches can be used
- optimizer
- loss function

### Two-way callbacks

The training loop also supports two-way callbacks. See the [FluxTraining.jl docs](https://fluxml.ai/FluxTraining.jl/dev/docs/callbacks/reference.md.html) for a list of all available callbacks. While supporting all the functionality of fastai's callbacks and training loop, it also provides [an extensible training loop API](https://fluxml.ai/FluxTraining.jl/dev/docs/tutorials/training.md.html) that makes it straightforward to integrate custom training steps with the available callbacks. As a result, different training steps for problems other than standard supervised training can make use of existing callbacks  without the need to handle control flow through callbacks. Additionally, callbacks have an additional level of safety by being required to declare what state they access and modify. With a little more effort up-front, this guarantees correct ordering of callback execution through [a dependency graph](https://fluxml.ai/FluxTraining.jl/dev/docs/callbacks/tipstricks.md.html#visualize-the-callback-dependency-graph). In the future, this will also make it possible to automatically run callbacks in parallel and asynchronously to reduce overhead by long-running callbacks like costly metric calculations and logging over the network.

### Encodings and blocks

In the paper, this subsection is in the low-level section (named Transforms and Pipelines), but I'm putting it here since it is the core of FastAI.jl's data block API. FastAI.jl provides `Encoding`s and `Block`s which correspond to fastai's `Transform`s and `Block`s. Encodings implement an `encode` (and optionally `decode`) function that describes how data corresponding to some blocks is transformed and how that transformation can be inverted. There is also support for stateful encodings like [`ProjectiveTransforms`](#) which need to use the same random state to augment every data point. Additionally, encodings describe what kind of block data is returned from encoding, allowing inspection of the whole data pipeline. The `Block`s are used to dispatch in the `encode` function to implement block-specific transformations. If no `encode` task is implemented for a pair of encoding and block, the default is to pass the data through unchanged like in fastai.

The `Block`s also allow implementing task-specific functionality:

- [`blocklossfn`](#) takes a prediction and encoded target block to determine a good loss function to use. For example, for image classification we want to compare two one-hot encoded labels and hence define `blocklossfn(::OneHotTensor{0}, ::OneHotTensor{0}) = logitcrossentropy`.
- [`blockmodel`](#) constructs a model from a backbone that maps an input block to an output block. For example, for image segmentation we have `ImageTensor{N}()` as the input block and `OneHotTensor{N}` (one-hot encoded N-dimensional masks) as output, so `blockmodel` turns the backbone into a U-Net.
- [`showblock!`](#) defines how to visualize a block of data.

### Generic optimizer

FastAI.jl uses the optimizers from Flux.jl, which provides a similarly [composable API for optimzers](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Composing-Optimisers).

### Generalized metric API

Metrics are handled by the [`Metrics`](#) callback which takes in reducing metric functions or [`FluxTraining.AbstractMetric`](#)s which have a similar API to fastai's.

### fastai.data.external

FastAI.jl makes all the same datasets available in `fastai.data.external` available. See [`datasets`](#) for a list of all datasets that can be downloaded.

### funcs_kwargs and DataLoader, fastai.data.core

In FastAI.jl, you are not restricted to a specific type of data iterator and can pass any iterator over batches to `Learner`. In cases where performance is important [`DataLoader`](#) can speed up data iteration by loading and batching samples in parallel on background threads. All transformations of data happen through the data container interface which requires a type to implement `Base.getindex`/`MLUtils.getobs` and `Base.length`/`MLUtils.numobs`, similar to PyTorch's `torch.utils.data.Dataset`. Data containers are then transformed into other data containers. Some examples:

- [`mapobs`](#)`(f, data)` lazily maps a function `f` of over `data` such that `getobs(mapobs(f, data), idx) == f(getobs(data, idx))`. For example `mapobs(loadfile, files)` turns a vector of image files into a data container of images.
- `DataLoader(data; batchsize)` is a wrapper around [`BatchView`](#) which turns a data container of samples into one of collated batches and `eachobsparallel` which creates a parallel, buffered iterator over the observations (here batches) in the resulting container.
- [`groupobs`](#)`(f, data)` splits a container into groups using a grouping function `f`. For example, `groupobs(grandparentname, files)` creates training splits for files where the grandparent folder indicates the split.
- [`MLUtils.ObsView`](#)`(data, idxs)` lazily takes a subset of the observations in `data`.

For more information, see the [data container tutorial](data_containers.md) and the [MLUtils.jl docs](https://juliaml.github.io/MLUtils.jl/dev/). At a higher level, there are also convenience functions like `loadfolderdata` to create data containers.

### Layers and architectures 

Flux.jl already does a better job at functionally creating model architectures than PyTorch, so FastAI.jl makes use of its layers. For example [`Flux.SkipConnection`](#)  corresponds to fastai's `MergeLayer`. The `FastAI.Models` submodule currently provides some high-level architectures like [`xresnet18`](#) and a U-Net builder [`UNetDynamic`](#) that can create U-Nets from *any* convolutional feature extractor. The [optional dependency](setup.md) [Metalhead.jl](https://github.com/FluxML/Metalhead.jl) also provides common pretrained vision models.


## Low-level APIs

Due to the nature of the Julia language and its design around multiple dispatch, packages tend to compose really well, so it was not necessary to reimplement or provide a unified API for low-level operations. We'll comment on the libraries that we were able to use.

### PyTorch foundations

Unlike Python, Julia has native support for N-dimensional regular arrays. As such, there is a standard interface for arrays and libraries don't need to implement their own. Consider that every deep learning framework in Python implements their own CPU and GPU arrays, which is part of the reason they are *frameworks*, not *libraries* (with the latter being vastly preferable). Julia's standard libraries implements the standard CPU `Array` type. GPU arrays are implemented through [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) `CuArray` type (with unified support for GPU vendors other than nvidia in the works). As a result, Flux.jl, the deep learning library of choice for FastAI.jl, does not need to reimplement their own CPU and GPU array versions. This kind of composability in general largely benefits what can be accomplished in Julia.

Some other libraries which are used under the hood: for image processing, the [Images.jl](https://juliaimages.org/) ecosystem of packages is used; for reading and processing tabular data [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) and [Tables.jl](https://github.com/JuliaData/Tables.jl); for plotting [Makie.jl](https://github.com/JuliaPlots/Makie.jl).

### Type dispatch

Multiple dispatch already is a core feature of the Julia language, hence the extensible interfaces in FastAI.jl are built around it and are natural fit for the language.

### Object-oriented semantic tensors

As mentioned above, Julia has great support for arrays with extra functionality available to packages that provide wrapper arrays like [NamedDims.jl](https://github.com/invenia/NamedDims.jl) which should generally *just work* with every part of the library. Hence there is no need for an addtional API that unifies separate packages, which in turn makes FastAI.jl more composable with other packages.

In encodings, the array types are used for dispatch only where an especially performant implementation is possible, and the block information is used for dispatching the semantics of the encoding.

### GPU-accelerated augmentation

FastAI.jl does not support GPU-accelerated augmentation (yet). Please open an issue if you run into a situation where data processing [becomes the bottleneck](background/datapipelines.md) and we'll prioritize this. The affine transformations implemented in DataAugmentation.jl and used in FastAI.jl are properly composed to ensure high quality results. They are also optimized for speed and memory usage (with complete support for inplace transformations).

### Convenience functionality

Much of the convenience provided by fastai is not required in Julia:

- `@delegates`: Due to the absence of deep class hierarchies, keyword arguments are seldom passed around (the only instance where this happens in FastAI.jl is [`tasklearner`](#)).
- `@patch`: since Julia is built around multiple dispatch, not classes, you just implement the task for a type, no patching needed
- `L`: due to first-class array support such a wrapper list container isn't needed

## nbdev

There is no `nbdev`-equivalent in Julia at the moment. That said, this documentation is generated by a document creation package [Pollen.jl](https://github.com/lorenzoh/Pollen.jl) that could be extended to support such a workflow. It already has support for different source and output formats like Jupyter notebooks, code execution and is built for interactive work with incremental rebuilds.

---

Hopefully this page has given you some context for how FastAI.jl relates to fastai and how to map concepts between the two. You are encouraged to go through the tutorials to see the design decisions made in practice.