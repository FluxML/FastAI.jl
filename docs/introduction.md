# Introduction

*This tutorial explains the qickstart examples and some core abstractions FastAI.jl is built on.*

{cell=main style="display:none;" result=false}
```julia
using FastAI
import FastAI: Image
```

On the [quickstart page](../notebooks/quickstart.ipynb), we showed how to train models on common tasks in a few lines of code like these:

```julia
using FastAI
data, blocks = loaddataset("imagenette2-160", (Image, Label))
task = ImageClassificationSingle(blocks)
learner = tasklearner(task, data, callbacks=[ToGPU()])
fitonecycle!(learner, 10)
showoutputs(task, learner)
```

Each of the five lines encapsulates one part of the deep learning pipeline to give a high-level API while still allowing customization. Let's have a closer look. 

## Dataset

{cell=main, result=false, output=false style="display:none;"}
```julia
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
```

{cell=main, output=false}
```julia
data, blocks = loaddataset("imagenette2-160", (Image, Label))
```

This line downloads and loads the [ImageNette](https://github.com/fastai/imagenette) image classification dataset, a small subset of ImageNet with 10 different classes. `data` is a [data container](data_containers.md) that can be used to load individual observations, here of images and the corresponding labels. We can use `getobs(data, i)` to load the `i`-th observation and `nobs` to find out how many observations there are.

{cell=main }
```julia
image, class = sample =  getobs(data, 1000)
@show class
image
```

`blocks` describe the format of the data that you want to use for learning. For supervised training tasks, they are a tuple of `(inputblock, targetblock)`. Since we want to do image classification, the input block is `Image{2}()`, representing a 2-dimensional image and the target block is `Label(classes)`, representing the class the image belongs to.

{cell=main}
```julia
blocks
```

## Learning task

{cell=main}
```julia
task = ImageClassificationSingle(blocks)
```

The next line defines a learning task which encapsulates the data preprocessing pipeline and other logic related to the task. `ImageClassificationSingle` is a simple wrapper around `BlockTask` which takes in blocks and data processing steps, so-called _encodings_. Using it, we can replace the above line with


```julia
task = BlockTask(
    (Image{2}(), Label(classes)),
    (
        ProjectiveTransforms((128, 128)),
        ImagePreprocessing(),
        OneHot()
    )
)
```

Based on the blocks and encodings, the learning task can derive lots of functionality:

- data processing
- visualization
- constructing task-specific models from a backbone
- creating a loss function

## Learner

{cell=main}
```julia
learner = tasklearner(task, data, callbacks=[ToGPU(), Metrics(accuracy)])
```

Next we create a [`Learner`](#) that encapsulates everything needed for training, including:
- parallelized training and validation data loaders using [`taskdataloaders`](#)
- a loss function using [`tasklossfn`](#)
- a task-specific model using [`taskmodel`](#)

The customizable, expanded version of the code looks like this:

```julia
dls = taskdataloaders(data, task)
model = taskmodel(task, Models.xresnet18())
lossfn = tasklossfn(task)
learner = Learner(model, dls, ADAM(), lossfn, ToGPU(), Metrics(accuracy))
```

At this step, we can also pass in any number of [callbacks](https://fluxml.ai/FluxTraining.jl/dev/docs/callbacks/reference.md.html) to customize the training. Here [`ToGPU`](#) ensures an available GPU is used, and [`Metrics`](#) adds additional metrics to track during training.

## Training

```julia
fitonecycle!(learner, 10)
```

Training now is quite simple. You have several options for high-level training schedules:

- [`lrfind`](#) to run a learning rate finder
- [`finetune!`](#) for when you're using a pretrained backbone
- [`fitonecycle!`](#) for when you're training a model from scratch



## Visualization

```julia
showoutputs(task, learner)
```

Finally, the last line visualizes the predictions of the trained model. It takes some samples from the training data loader, runs them through the model and decodes the outputs. How each piece of data is visualized is also inferred through the blocks in the learning task.