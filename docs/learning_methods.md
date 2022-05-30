# Custom learning tasks

*This tutorial explains the low-level interface behind `BlockTask`s and how to use it to create your custom learning tasks without the data block interface.*

In the [quickstart](quickstart.md) section, you've already seen a learning task in action: [`BlockTask`](#). The learning task abstraction powers FastAI.jl's high-level interface allowing you to make training models for a task simple. `BlockTask` is a particularly convenient and composable interface for creating learning tasks and should be preferred for most use cases.

However, to get a look behind the scenes, in this tutorial we'll use the lower-level learning task interface to implement our own version of an image classification learning task. You're encouraged to follow along in a REPL or notebook. This tutorial can also serve as a template for implementing a custom learning task for your own project.

A learning task describes how we need to process data so we can train a model for some task. In our case, the task we want to solve is to classify an image. The task defines what kind of data we need, here pairs of images and class labels. That alone, however, isn't enough to train a model since we can't just throw an image in any format into a model and get a class out. Almost always the input data needs to be processed in some way before it is input to a model (we call this **encoding**) and the same goes for the model outputs (we call this **decoding**).

So let's say we have an image and a trained model. How do we make a prediction? First we encode the image, run it through the model, and then decode the output. Similarly, how we can use a pair of image and class to train a model? We encode both, run the encoded input through the model and then compare the output with the encoded class using a **loss function**. The result tells us how we'll need to update the weights of the model to improve its performance.

In essence, the learning task interface allows us to implement these steps and derive useful functionality from it, like training and evaluating models. Later we'll also cover some optional interfaces that allow us to define other parts of a deep learning project.

## Datasets

Before we get started, let's load up a [data container](data_containers.md) that we can test our code on as we go. It's always a good idea to interactively test your code! Since we'll be implementing a task for image classification, the observations in our data container will of course have to be pairs of images and classes. We'll use one of the many image classification datasets available from the fastai dataset repository. I'll use ImageNette, but you can use any of the datasets listed in `FastAI.Datasets.DATASETS_IMAGECLASSIFICATION`. The way the interface is built allows you to easily swap out the dataset you're using.

{cell=main}
```julia
using FastAI, FastAI.DataAugmentation, Colors
import FastAI: Image
data = Datasets.loadfolderdata(
    load(datasets()["imagenette2-160"]),
    filterfn=isimagefile,
    loadfn=(loadfile, parentname))
```

We'll also collect the unique class names:

{cell=main}
```julia
images, targets = data
classes = unique(eachobs(targets))
```

## Implementation

### Learning task struct

Now let's get to it! The first thing we need to do is to create a [`LearningTask`](#) struct. The `LearningTask` `struct` should contain all the configuration needed for encoding and decoding the data. We'll keep it simple here and include a list of the classes and the image dimensions input to the model.

{cell=main}
```julia
struct ImageClassification <: FastAI.LearningTask
    classes
    size
end
```

Now we can create an instance of it, though of course it can't do anything (yet!).

{cell=main, result=true}
```julia
task = ImageClassification(classes, (128, 128))
```

### Encoding and decoding

There are 3 tasks we need to define before we can use our learning task to train models and make predictions:

- [`encodesample`](#) which encodes an image and a class
- [`encodeinput`](#) will encode an image so it can be input to a model
- [`decodeypred`](#) decodes a model output into a class label

Note: These functions always operate on *single* images and classes, even if we want to pass batches to the model later on.

While it's not the focus of this tutorial, let's give a quick recap of how the data is encoded and decoded for image classification.

- Images are cropped to a common size so they can be batched, converted to a 3D array with dimensions (height, width, color channels) and normalized
- Classes are encoded as one-hot vectors, teaching the model to predict a confidence distribution over all classes. To decode a predicted one-hot vector, we can simply find the index with the highest value and look up the class label.

Each of the tasks also takes a `context::`[`FastAI.Context`](#) argument which allows it to behave differently during training, validation and inference. We'll make use of that to choose a different image crop for each situation. During training we'll use a random crop for augmentation, while during validation a center crop will ensure that any metrics we track are the same every epoch. During inference, we won't crop the image so we don't lose any information.

#### Inputs

We implement [`encodeinput`](#) using [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl). Feel free to look at [its documentation](https://lorenzoh.github.io/DataAugmentation.jl/dev/docs/literate/intro.html), we won't focus on it here.

{cell=main}
```julia
using FastAI.Vision: IMAGENET_MEANS, IMAGENET_STDS  # color statistics for normalization

# Helper for crop based on context
getresizecrop(context::Training, sz) = DataAugmentation.RandomResizeCrop(sz)
getresizecrop(context::Validation, sz) = CenterResizeCrop(sz)
getresizecrop(context::Inference, sz) = ResizePadDivisible(sz, 32)

function FastAI.encodeinput(
        task::ImageClassification,
        context::Context,
        image)
    tfm = DataAugmentation.compose(
        getresizecrop(context, task.size),
        ToEltype(RGB{Float32}),
        ImageToTensor(),
        Normalize(IMAGENET_MEANS, IMAGENET_STDS);
    )
    return apply(tfm, DataAugmentation.Image(image)) |> itemdata
end
```

If we test this out on an image, it should give us a 3D array of size `(128, 128, 3)`, and indeed it does:

{cell=main}
```julia
sample = image, class = getobs(data, 1)
x = FastAI.encodeinput(task, Training(), image)
summary(x)
```

#### Outputs

`encodetarget` is much simpler:

{cell=main}
```julia
function FastAI.encodetarget(
        task::ImageClassification,
        ::Context,
        class)
    idx = findfirst(isequal(class), task.classes)
    v = zeros(Float32, length(task.classes))
    v[idx] = 1.
    return v
end

FastAI.encodesample(task::ImageClassification, ctx, (input, target)) = (
    encodeinput(task, ctx, input),
    encodetarget(task, ctx, target),
)


```

{cell=main}
```julia
y = FastAI.encodetarget(task, Training(), class)
```

The same goes for the decoding step:

{cell=main}
```julia
function FastAI.decodeypred(task::ImageClassification, ::Context, ypred)
    return task.classes[argmax(ypred)]
end
```

{cell=main}
```julia
FastAI.decodeypred(task, Training(), y) == class
```

## Training

And that's all we need to start training models! There are some optional interfaces that make that even easier, but let's use what we have for now.

With our `LearningTask` defined, we can use [`taskdataloaders`](#) to turn a dataset into a set of training and validation data loaders that can be thrown into a training loop.

{cell=main}
```julia
traindl, valdl = taskdataloaders(data, task)
```

Now, with a makeshift model, an optimizer and a loss function we can create a [`Learner`](#).

{cell=main}
```julia
using FastAI, Flux

model = Chain(
    Models.xresnet18(),
    Chain(
            AdaptiveMeanPool((1,1)),
            Flux.flatten,
            Dense(512, length(task.classes)),
    )
)
opt = ADAM()
lossfn = Flux.Losses.logitcrossentropy

learner = Learner(model, (traindl, valdl), opt, lossfn)
```

From here, you're free to start training using  [`fit!`](#) or [`fitonecycle!`](#).

These tasks are also enough to use [`predict`](#) and [`predictbatch`](#) once you've trained a model.

## Additional interfaces

### Training interface

We can implement some additional tasks to make our life easier. Specifically, let's implement every task needed to use [`tasklearner`](#):

- [`tasklossfn`](#): return a loss function `lossfn(ys, ys)` comparing a batch of model outputs and encoded targets
- [`taskmodel`](#): from a backbone, construct a model suitable for the task

Let's start with the loss function. We want to compare two one-hot encoded categorical variables, for which categorical cross entropy is the most commonly used loss function.

{cell=main}
```
FastAI.tasklossfn(task::ImageClassification) = Flux.Losses.logitcrossentropy
```

For the model, we'll assume we're getting a convolutional feature extractor passed in as a backbone so its output will be of size (height, width, channels, batch size). [`Flux.outputsize`](#) can be used to calculate the output size of arbitrary models without having to evaluate the model. We'll use it to check the number of output channels of the backbone. Then we add a global pooling layer and some dense layers on top to get a classification output. 

{cell=main}
```julia
function FastAI.taskmodel(task::ImageClassification, backbone)
    h, w, outch, b = Flux.outputsize(backbone, (256, 256, inblock.nchannels, 1))
    head = Chain(
        AdaptiveMeanPool((1, 1)),
        Dense(outch, 512),
        BatchNorm(512),
        Dense(512, length(task.classes))
    )
    return Chain(backbone, head)
end
```
