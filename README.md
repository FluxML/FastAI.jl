# FastAI

A first cut at a port of the FastAI V2 API to Julia

This code is inspired by FastAI, but differs in implementation
in several ways.  Most importantly, the original Python code
makes heavy use of side-effects where the Learner holds different
state variables, and other objects access and modify them.

This has been replaced by a more functional design.  The state
is now transmitted via arguments to Callbacks which may then pass them
on to Metrics

Much of the documentation has been copied from the original Python,
and modified where appropriate

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://opus111.github.io/FastAI.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://opus111.github.io/FastAI.jl/dev)
[![Build Status](https://travis-ci.com/opus111/FastAI.jl.svg?branch=master)](https://travis-ci.com/opus111/FastAI.jl)
