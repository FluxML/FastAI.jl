# Introduction

*This tutorial explains the qickstart examples and some core abstractions FastAI.jl is built on.*

{cell=main style="display:none;" result=false}
```julia
using FastAI
```

On the [quickstart page](quickstart.ipynb), we showed how to train models on common tasks in a few lines of code:

```julia
using FastAI

path = datasetpath("imagenette2-160")
dataset = loadtaskdata(path, ImageClassificationTask)
method = ImageClassification(Datasets.getclassesclassification("imagenette2-160"), (160, 160))
dls = methoddataloaders(dataset, method, 16)
model = methodmodel(method, Models.xresnet18())
learner = Learner(model, dls, ADAM(), methodlossfn(method), ToGPU(), Metrics(accuracy))
fitonecycle!(learner, 5)
```

Let's unpack each line.

## Data containers

{cell=main}
```julia
path = datasetpath("imagenette2-160")
dataset = loadtaskdata(path, ImageClassificationTask)
```

These two lines download and load the [ImageNette](https://github.com/fastai/imagenette) image classification dataset, a small subset of ImageNet with 10 different classes. `dataset` is a [data container](data_containers.md) that can be used to load individual observations, here of images and the corresponding labels. We can use `getobs(dataset, i)` to load the `i`-th observation and `nobs` to find out how many observations there are.

{cell=main }
```julia
image, class = getobs(dataset, 1000)
@show class
image
```

{cell=main}
```julia
nobs(dataset)
```

To train on a different dataset, you could replace `dataset` with other data containers made up of pairs of images and classes.

## Method

{cell=main}
```julia
classes = Datasets.getclassesclassification("imagenette2-160")
method = ImageClassification(classes, (160, 160))
```

Here we define an instance of the learning method [`ImageClassification`](#) which defines how data is processed before being fed to the model and how model outputs are turned into predictions. `classes` is a vector of strings naming each class, and `(224, 224)` the size of the images that are input to the model.

`ImageClassification` is a `LearningMethod`, an abstraction that encapsulates the logic and configuration for training models on a specific learning task. See [learning methods](learning_methods.md) to find out more about how they can be used and how to create custom learning methods.

## Data loaders

{cell=main}
```julia
dls = methoddataloaders(dataset, method, 16)
```

Next we turn the data container into training and validation data loaders. These take care of efficiently loading batches of data (by default in parallel). The observations are already preprocessed using the information in `method` and then batched together. Let's look at a single batch:

{cell=main}
```julia
traindl, valdl = dls
(xs, ys), _ = iterate(traindl)
summary.((xs, ys))
```

`xs` is a batch of cropped and normalized images with dimensions `(height, width, color channels, batch size)` and `ys` a batch of one-hot encoded classes with dimensions `(classes, batch size)`.

## Model

{cell=main}
```julia
model = methodmodel(method, Models.xresnet18())
```

Now we create a Flux.jl model. [`methodmodel`](#) is a part of the learning method interface that knows how to smartly construct an image classification model from different backbone architectures. Here a classificiation head with the appropriate number of classes is stacked on a slightly modified version of the ResNet architecture.

## Learner

{cell=main}
```julia
learner = Learner(model, dls, ADAM(), methodlossfn(method), ToGPU(), Metrics(accuracy))
```

Finally we bring the model and data loaders together with an optimizer and loss function in a `Learner`. The `Learner` stores all state for training the model. It also features a powerful, extensible [callback system](https://lorenzoh.github.io/FluxTraining.jl/dev/docs/callbacks/reference.html) enabling checkpointing, hyperparameter scheduling, TensorBoard logging, and many other features. Here we use the `ToGPU()` callback so that model and batch data will be transferred to an available GPU and `Metrics(accuracy)` to track the classification accuracy during training.

With that setup, training `learner` is dead simple:

```julia
fitonecycle!(learner, 5)
```



