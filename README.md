# FastAI.jl

![](fastai-julia-logo.png)
FastAI.jl is a Julia library for training state-of-the art deep learning models.

From loading datasets and creating data preprocessing pipelines to training, FastAI.jl takes the boilerplate out of deep learning projects. It equips you with reusable components for every part of your project while remaining customizable at every layer. FastAI.jl comes with support for common computer vision and tabular data learning tasks, with more to come.

FastAI.jl's high-level workflows combine functionality from many packages in the ecosystem, most notably [Flux.jl](https://github.com/FluxML/Flux.jl), [FluxTraining.jl](https://github.com/FluxML/FluxTraining.jl), [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl) and [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl).

See our [**documentation**](https://fluxml.ai/FastAI.jl) to find out more.

## Example

As an example, here is how to train an image classification model:

```julia
using FastAI
data, blocks = load(datarecipes()["imagenette2-320"])
task = ImageClassificationSingle(blocks)
learner = tasklearner(task, data, callbacks=[ToGPU()])
fitonecycle!(learner, 10)
showoutputs(task, learner)
```

## Setup

To get started, install FastAI.jl using the Julia package manager: 

```julia
using Pkg
Pkg.add("FastAI")
```

or try it out with this [Google Colab template](https://colab.research.google.com/gist/lorenzoh/2fdc91f9e42a15e633861c640c68e5e8).

## Getting started

To dive in, you may be interested in

- an [overview of the high-level API](https://fluxml.ai/FastAI.jl/dev/documents%2Fdocs%2Fintroduction.md),
- seeing some [example learning tasks](https://fluxml.ai/FastAI.jl/dev/documents%2Fnotebooks%2Fquickstart.ipynb),
- finding out [how you can search for and find datasets and other functionality](https://fluxml.ai/FastAI.jl/dev/documents%2Fdocs%2Fdiscovery.md); or
- [our contributor guide](CONTRIBUTING.md)

## Get in touch

You can get in touch here on GitHub or on the JuliaLang Zulip in the [`#ml-contributors` channel](https://julialang.zulipchat.com/#narrow/stream/237432-ml-contributers).

---
## Acknowledgements

FastAI.jl takes inspiration from the fantastic [fastai](http://docs.fast.ai) library for Python. Jeremy Howard and the fastai team kindly approved this project and its use of the fastai name.

This project also builds on many packages in the Julia ecosystem.
