# FastAI
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://FluxML.github.io/FastAI.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://FluxML.github.io/FastAI.jl/dev)
[![Build Status](https://github.com/FluxML/FastAI.jl/workflows/CI/badge.svg)](https://github.com/FluxML/FastAI.jl/actions)

![Logo](https://raw.githubusercontent.com/opus111/FastAI.jl/master/fastai-julia-logo.png)

FastAI.jl is inspired by [fastai](https://github.com/fastai/fastai/blob/master/fastai/), and is a repository of best practices for Deep Learning with Flux. Its goal is to enable creating state-of-the-art models, while freeing the developer from having to implement most of the sub-components. FastAI allows the design, training, and delivery of models that compete with the best in class, using a few lines of code.

FastAI.jl contains thorough documentation, examples and tutorials, but does not contain the source of the core components.  It is an umbrella package combining the functionality specialized packages.  These packages include:

- Flux.jl: 100% pure-Julia Deep Learning stack. Provides lightweight abstractions on top of Julia's native GPU and AD support.

- FluxTraining.jl: Enables eaily customized training loops, a large library of useful metrics, and many useful utilities

- FluxONNX.jl: Support for reading and writing models in the ONNX file format.  ONNX enables Transfer Learning by enabling the import of very large pretrained models.

- MLDataPattern.jl: Utility package for subsetting, partitioning, iterating, and resampling of Machine Learning datasets.

- MLDataSets.jl: A community effort to provide a common interface for accessing common Machine Learning (ML) datasets.

- DataAugmentation.jl: Utilities for augmenting image data

- Metalhead.jl: Computer vision models for Flux 

- Transformers.jl: Implementation of NLP transformer-based models


*Note*: this is a package in-development. It is not ready for use, and one should expect major breaking changes for the foreseeable future. However, we are very interested in meeting the desires of the community, so all comments and contributions are welcome via PRs/issues.
