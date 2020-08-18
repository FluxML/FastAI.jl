# FastAI
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://FluxML.github.io/FastAI.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://FluxML.github.io/FastAI.jl/dev)
[![Build Status](https://github.com/FluxML/FastAI.jl/workflows/CI/badge.svg)](https://github.com/FluxML/FastAI.jl/actions)

![Logo](https://raw.githubusercontent.com/opus111/FastAI.jl/master/fastai-julia-logo.png)

This code is inspired by [fastai](https://github.com/fastai/fastai/blob/master/fastai/), but differs in implementation in several ways. Most importantly, the original Python code makes heavy use of side-effects where the `Learner` holds different state variables, and other objects access and modify them.

This has been replaced by a more functional design. The state is now transmitted via arguments to `Callbacks` which may then pass them on to `Metrics`.

*Note*: this is a package in-development. Expect breaking changes for the foreseeable future, but we want you to test out the package by following the documentation. Any contributions are welcome via PRs/issues.

Much of the documentation has been copied from the original Python, and modified where appropriate.