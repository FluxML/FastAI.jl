# FastAI

A port of fastai v2 to Julia.

![Logo](https://raw.githubusercontent.com/opus111/FastAI.jl/master/fastai-julia-logo.png)

This code is inspired by [fastai](https://github.com/fastai/fastai/blob/master/fastai/), but differs in implementation in several ways. Most importantly, the original Python code makes heavy use of side-effects where the `Learner` holds different state variables, and other objects access and modify them.

This has been replaced by a more functional design. The state is now transmitted via arguments to `Callbacks` which may then pass them on to `Metrics`.

*Note*: this is a package in-development. Expect breaking changes for the foreseeable future, but we want you to test out the package by following the documentation. Any contributions are welcome via PRs/issues.

Much of the documentation has been copied from the original Python, and modified where appropriate.

The original source is can be found at [https://github.com/fastai/fastai/blob/master/fastai/](
https://github.com/fastai/fastai/blob/master/fastai/). The original documentation can be found at [http://docs.fast.ai](http://docs.fast.ai).