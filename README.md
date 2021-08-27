# FastAI

[Documentation](https://FluxML.github.io/FastAI.jl/dev)

FastAI.jl is inspired by [fastai](https://github.com/fastai/fastai), and is a repository of best practices for deep learning in Julia. Its goal is to easily enable creating state-of-the-art models. FastAI enables the design, training, and delivery of deep learning models that compete with the best in class, using few lines of code.

Install with

```julia
using Pkg
Pkg.add("FastAI")
```

or try it out with this [Google Colab template](https://colab.research.google.com/gist/lorenzoh/2fdc91f9e42a15e633861c640c68e5e8).


As an example, here is how to train an image classification model:

```julia
using FastAI
data, blocks = loaddataset("imagenette2-160", (Image, Label))
method = ImageClassificationSingle(blocks)
learner = methodlearner(method, data, Models.xresnet18(), ToGPU())
fitonecycle!(learner, 10)
plotpredictions(method, learner)
```

Please read [the documentation](https://fluxml.github.io/FastAI.jl/dev) for more information and see the [setup instructions](docs/setup.md).
