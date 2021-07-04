# Learning methods

*This tutorial explains what learning tasks and methods are and how to create your own.*

In the [quickstart](quickstart.md) section, you've already seen a learning method in action: [`ImageClassification`](#). The learning method abstraction powers FastAI.jl's high-level interface allowing you to make training models for a task simple. In this tutorial we'll implement our own version of the image classification learning method. You're encouraged to follow along in a REPL or notebook. This tutorial can also serve as a template for implementing a custom learning method for your own project.

A learning method describes how we need to process data so we can train a model for some task. In our case, the task we want to solve is to classify an image. The task defines what kind of data we need, here pairs of images and class labels. That alone, however, isn't enough to train a model since we can't just throw an image in any format into a model and get a class out. Almost always the input data needs to be processed in some way before it is input to a model (we call this **encoding**) and the same goes for the model outputs (we call this **decoding**).

So let's say we have an image and a trained model. How do we make a prediction? First we encode the image, run it through the model, and then decode the output. Similarly, how we can use a pair of image and class to train a model? We encode both, run the encoded input through the model and then compare the output with the encoded class using a **loss function**. The result tells us how we'll need to update the weights of the model to improve its performance.

In essence, the learning method interface allows us to implement these steps and derive useful functionality from it, like training and evaluating models. Later we'll also cover some optional interfaces that allow us to define other parts of a deep learning project.

## Setup

Next to FastAI.jl, you'll need to install

```juliarepl
] add DataAugmentation DLPipelines Colors
```

## Datasets

Before we get started, let's load up a [data container](data_containers.md) that we can test our code on as we go. It's always a good idea to interactively test your code! Since we'll be implementing a method for image classification, the observations in our data container will of course have to be pairs of images and classes. We'll use one of the many image classification datasets available from the fastai dataset repository. I'll use ImageNette, but you can use any of the datasets listed in `FastAI.Datasets.DATASETS_IMAGECLASSIFICATION`. The way the interface is built allows you to easily swap out the dataset you're using.

{cell=main}
```julia
using FastAI
using FastAI.Datasets
DATASET = "imagenette2-160"
data = Datasets.loadtaskdata(Datasets.datasetpath(DATASET), ImageClassification)
image, class = getobs(data, 1)
image
```

We'll also collect the unique class names:

{cell=main}
```julia
classes = unique([getobs(data.target, i) for i in 1:nobs(data.target)])
```



## Implementation

### Learning method struct

Now let's get to it! The first thing we need to do is to create a [`DLPipelines.LearningMethod`](#) struct. The `LearningMethod` `struct` should contain all the configuration needed for encoding and decoding the data. We'll keep it simple here and include a list of the classes and the image dimensions input to the model. The reference implementation [`ImageClassification`](#) of course has many more parameters that can be configured.

{cell=main}
```julia
using FastAI: DLPipelines

struct MyImageClassification <: DLPipelines.LearningMethod
    classes
    size
end
```

Now we can create an instance of it, though of course it can't do anything (yet!).

{cell=main, result=false}
```julia
method = MyImageClassification(classes, (128, 128))
```

### Encoding and decoding

There are 3 methods we need to define before we can use our learning method to train models and make predictions:

- `DLPipelines.encodeinput` will encode an image so it can be input to a model;
- `DLPipelines.encodetarget` encodes a class so we can compare it with a model output; and
- `DLPipelines.decodeŷ` (write `\hat<TAB>` for  ` ̂`) decodes a model output into a class label

Note: These functions always operate on *single* images and classes, even if we want to pass batches to the model later on.

While it's not the focus of this tutorial, let's give a quick recap of how the data is encoded and decoded for image classification.

- Images are cropped to a common size so they can be batched, converted to a 3D array with dimensions (height, width, color channels) and normalized
- Classes are encoded as one-hot vectors, teaching the model to predict a confidence distribution over all classes. To decode a predicted one-hot vector, we can simply find the index with the highest value and look up the class label.

Each of the methods also takes a `context::`[`DLPipelines.Context`](#) argument which allows it to behave differently during training, validation and inference. We'll make use of that to choose a different image crop for each situation. During training we'll use a random crop for augmentation, while during validation a center crop will ensure that any metrics we track are the same every epoch. During inference, we won't crop the image so we don't lose any information.

#### Inputs

We implement [`encodeinput`](#) using [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl). Feel free to look at [its documentation](https://lorenzoh.github.io/DataAugmentation.jl/dev/docs/literate/intro.html), we won't focus on it here.

{cell=main}
```julia
using DataAugmentation
using Colors: RGB
using FastAI: IMAGENET_MEANS, IMAGENET_STDS  # color statistics for normalization

# Helper for crop based on context
getresizecrop(context::Training, sz) = DataAugmentation.RandomResizeCrop(sz)
getresizecrop(context::Validation, sz) = CenterResizeCrop(sz)
getresizecrop(context::Inference, sz) = ResizePadDivisible(sz, 32)

function DLPipelines.encodeinput(
        method::MyImageClassification,
        context::Context,
        image)
    tfm = DataAugmentation.compose(
        getresizecrop(context, method.size),
        ToEltype(RGB{Float32}),
        ImageToTensor(),
        Normalize(IMAGENET_MEANS, IMAGENET_STDS);
    )
    return apply(tfm, Image(image)) |> itemdata
end
```

If we test this out on an image, it should give us a 3D array of size `(128, 128, 3)`, and indeed it does:

{cell=main}
```julia
x = encodeinput(method, Training(), image)
summary(x)
```

#### Outputs

`encodetarget` is much simpler:

{cell=main}
```julia
function DLPipelines.encodetarget(
        method::MyImageClassification,
        ::Context,
        class)
    idx = findfirst(isequal(class), method.classes)
    v = zeros(Float32, length(method.classes))
    v[idx] = 1.
    return v
end
```

{cell=main}
```julia
y = encodetarget(method, Training(), class)
```

The same goes for the decoding step:

{cell=main}
```julia
function DLPipelines.decodeŷ(method::MyImageClassification, ::Context, ŷ)
    return method.classes[argmax(ŷ)]
end
```

{cell=main}
```julia
decodeŷ(method, Training(), y) == class
```

## Training

And that's all we need to start training models! There are some optional interfaces that make that even easier, but let's use what we have for now.

With our `LearningMethod` defined, we can use [`methoddataloaders`](#) to turn a dataset into a set of training and validation data loaders that can be thrown into a training loop.

{cell=main}
```julia
traindl, valdl = methoddataloaders(data, method)
```

Now, with a makeshift model, an optimizer and a loss function we can create a [`Learner`](#).

{cell=main}
```julia
using FastAI: Flux

model = Chain(
    Models.xresnet18(),
    Chain(
            AdaptiveMeanPool((1,1)),
            flatten,
            Dense(512, length(method.classes)),
    )
)
opt = ADAM()
lossfn = Flux.Losses.logitcrossentropy

learner = Learner(model, (traindl, valdl), opt, lossfn)
```

From here, you're free to start training using  [`fit!`](#) or [`fitonecycle!`](#).