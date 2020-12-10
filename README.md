# FastAI
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://FluxML.github.io/FastAI.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://FluxML.github.io/FastAI.jl/dev)
[![Build Status](https://github.com/FluxML/FastAI.jl/workflows/CI/badge.svg)](https://github.com/FluxML/FastAI.jl/actions)

![Logo](https://raw.githubusercontent.com/opus111/FastAI.jl/master/fastai-julia-logo.png)

FastAI.jl is inspired by [fastai](https://github.com/fastai/fastai/blob/master/fastai/), and is a repository of best practices for Deep Learning with Flux. Its goal is to enable creating state-of-the-art models, while freeing the developer from having to implement most of the sub-components. FastAI allows the design, training, and delivery of models that compete with the best in class, using a few lines of code.

FastAI.jl contains thorough documentation, examples and tutorials, but does not contain the source of the core components.  It is an umbrella package combining the functionality specialized packages.  These packages include:
- [Flux.jl](https://github.com/FluxML/Flux.jl): 100% pure-Julia Deep Learning stack. Provides lightweight abstractions on top of Julia's core GPU and AD support.
- [FluxTraining.jl](https://github.com/lorenzoh/FluxTraining.jl): Easily customized training loops, a large library of useful metrics, and many useful utilities (such as logging)
- [DataLoaders.jl](https://github.com/lorenzoh/DataLoaders.jl): Multi-threaded data loading built on MLDataPattern.jl (similar to PyTorch's `DataLoader`).
- [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl): Utility package for subsetting, partitioning, iterating, and resampling of Machine Learning datasets.
- [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl): A community effort to provide a common interface for accessing common Machine Learning (ML) datasets.
- [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl): Utilities for augmenting image data
- [Metalhead.jl](https://github.com/FluxML/Metalhead.jl): Computer vision models for Flux 
- [Transformers.jl](https://github.com/chengchingwen/Transformers.jl): NLP transformer-based models for Flux

*Note*: this is a package in-development. One should expect major breaking changes for the foreseeable future. But we are very interested in meeting the desires of the community, so all comments and contributions are welcome via PRs/issues.
