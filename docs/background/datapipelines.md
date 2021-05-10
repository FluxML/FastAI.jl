# Performant data pipelines

*Bottlenecks in data pipelines and how to measure and fix them*

When training large deep learning models on a GPU we clearly want wait as short as possible for the training to complete. The hardware bottleneck is usually the GPU power you have available to you. This means that data pipelines need to be fast enough to keep the GPU at 100% utilization, that is, keep it from "starving". Reducing the time the GPU has to wait for the next batch of data directly lowers the training time until the GPU is fully utilized. There are other ways to reduce training time like using hyperparameter schedules and different optimizers for faster convergence, but we'll only talk about improving GPU utilization here.

## Reasons for low GPU utilization

The main cause of low GPU utilization is that the next batch of data is not available after a training step and the GPU has to wait. This means that in order to get full GPU utilization,

1. loading a batch must not take longer than a training step; and
2. the data must be loaded in the background, so that it is ready the moment the GPU needs it.

These issues can be addressed by

1. using worker threads to load multiple batches in parallel
2. keeping the primary thread free; and
3. reducing the time it takes to load a single batch

FastAI.jl by default uses `DataLoader` from the [DataLoaders.jl]() package which addresses points 1. and 2. For those familiar with PyTorch, it closely resembles `torch.utils.data.DataLoader`. It also efficiently collates the data by reusing a buffer where supported.

We can measure the large performance difference by comparing a naive sequential data iterator with `eachobsparallel`, the data iterator that `DataLoader` uses.

```julia
using DataLoaders: batchviewcollated
using FastAI
using FastAI.Datasets

data = loadtaskdata(datasetpath("imagenette2-320"), ImageClassificationTask)
method = ImageClassification(Datasets.getclassesclassification("imagenette2-320"), (224, 224))

# maps data processing over `data`
methoddata = methoddataset(data, method, Training())

# creates a data container of collated batches
batchdata = batchviewcollated(methoddata, 16)

NBATCHES = 200

# sequential data iterator
@time for (i, batch) in enumerate(getobs(batchdata, i) for i in 1:nobs(batchdata))
    i != NBATCHES || break
end

# parallel data iterator
@time for (i, batch) in enumerate(eachobsparallel(batchdata))
    i != NBATCHES || break
end
```

Running each timer twice to forego compilation time, the sequential iterator takes 20 seconds while the parallel iterator using 11 background threads only takes 2.5 seconds. This certainly isn't a proper benchmark, but it shows the performance can be improved by an order of magnitude with no effort.

Beside increasing the amount of compute available with worker threads as above, the data loading performance can also be improved by reducing the time it takes to load a single batch. Since a batch is made up of some number of observations, this usually boils down to reducing the loading time of a single observation. If you're using the `LearningMethod` API, this can be further broken down into the loading and encoding part.

## Measuring performance

So how do you know if your GPU is underutilized? If it isn't, then improving data pipeline performance won't help you at all! One way to check this is to start training and run `> watch -n 0.1 nvidia-smi` in a terminal which displays and refreshs GPU stats every 1/10th of a second. If `GPU-Util` stays between 90% and 99%, you're good! 

If that's not the case, you might see it frantically jumping up and down. We can get a better estimate of how much training time can be sped up by running the following experiment:

1. Load one batch and run `n` optimization steps on this batch. The time this takes corresponds to the training time when the GPU does not have to wait for data to be available.
2. Next take your data iterator and time iterating over the first `n` batches *without* an optimization step.

The speed of the complete training loop (data loading and optimization) will be around the maximum of either measurement. Roughly speaking, if 1. takes 100 seconds and 2. takes 200 seconds, you know that you can speed up training by about a factor of 2 if you reduce data loading time by half, after which the GPU will become the bottleneck.

```julia
using FastAI
using FastAI.Datasets
using FluxTraining: fitbatchphase!

data = loadtaskdata(datasetpath("imagenette2-320"), ImageClassificationTask)
method = ImageClassification(Datasets.getclassesclassification("imagenette2-320"), (224, 224))

learner = methodlearner(method, data, xresnet18())

NBATCHES = 100

# Measure GPU time
batch = gpu(first(learner.data.training))
learner.model = gpu(model)
@time for i in 1:NBATCHES
    fitbatchphase!(learner, batch, TrainingPhase())
end

# Measure data loading time
@time for (i, batch) in zip(learner.data.training, 1:NBATCHES)
end
```

Again, make sure to run each measurement twice so you don't include the compilation time.

---

To find performance bottlenecks in the loading of each observation, you'll want to compare the time it takes to load an observation of the task data container and the time it takes to encode that observation. 

```julia
using BenchmarkTools
using FastAI
using FastAI.Datasets

# Since loading times can vary per observation, we'll average the measurements over multiple observations
N = 10
data = datasubset(shuffleobs(loadtaskdata(datasetpath("imagenette2"), ImageClassificationTask), 1:N))
method = ImageClassification(Datasets.getclassesclassification("imagenette2-320"), (224, 224))

# Time it takes to load an `(image, class)` observation
@btime for i in 1:N
    getobs(data, i)
end


# Time it takes to encode an `(image, class)` observation into `(x, y)`
obss = [getobs(data, i) for i in 1:N]
@btime for i in 1:N
    encode(method, Training(), obss[i])
end
```

This will give you a pretty good idea of where the performance bottleneck is. Note that the encoding performance is often dependent of the method configuration. If we used `ImageClassification` with input size `(64, 64)` it would be much faster.

## Improving performance

So, you've identified the data pipeline as a performance bottleneck. What now? Before anything else, make sure you're doing the following:

- Use `DataLoaders.DataLoader` as a data iterator. If you're using [`methoddataloaders`](#) or [`methodlearner`](#), this is already the case.
- Start Julia with multiple threads by specifying the `-t n`/`-t auto` flag when starting Julia. If it is successful, `Threads.nthreads()` should be larger than `1`.

If the data loading is still slowing down training, you'll probably have to speed up the loading of each observation. As mentioned above, this can be broken down into observation loading and encoding. The exact strategy will depend on your use case, but here are some examples.

### Reduce loading time of image datasets by presizing 

For many computer vision tasks, you will resize and crop images to a specific size during training for GPU performance reasons. If the images themselves are large, loading them from disk itself can take some time. If your dataset consists of 1920x1080 resolution images but you're resizing them to 256x256 during training, you're wasting a lot of time loading the large images. *Presizing* means saving resized versions of each image to disk once, and then loading these smaller versions during training. We can see the performance difference using ImageNette since it comes in 3 sizes: original, 360px and 180px.

```julia
data_orig = loadtaskdata(datasetpath("imagenette2"), ImageClassificationTask)
@time eachobsparallel(data_orig, buffered = false)

data_320px = loadtaskdata(datasetpath("imagenette2-320"), ImageClassificationTask)
@time eachobsparallel(data_320px, buffered = false)

data_160px = loadtaskdata(datasetpath("imagenette2-160"), ImageClassificationTask)
@time eachobsparallel(data_160px, buffered = false)
```

### Reducing allocations with inplace operations

When implementing the `LearningMethod` interface, you have the option to implement `encode!(buf, method, context, sample)`, an inplace version of `encode` that reuses a buffer to avoid allocations. Reducing allocations often speeds up the encoding step and can also reduce the frequency of garbage collector pauses during training which can reduce GPU utilization.

### Using efficient data augmentation

Many kinds of augmentation can be composed efficiently. A prime example of this are image transformations like resizing, scaling and cropping which are powered by [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl). See [its documentation](https://lorenzoh.github.io/DataAugmentation.jl/dev/docs/literate/intro.html) to find out how to implement efficient, composable data transformations.