#=
FastAI.jl:

Author: Peter Wolf (opus111@gmail.com)

A first cut at a port of the FastAI V2 API to Julia

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#

module FastAI

using Random
using StatsBase
using Statistics
using Flux
using Flux: update!
using Flux.Data
using Base: length, getindex
using Random: randperm

export AbstractLearner
export AbstractCallback
export AbstractMetric
export IterableDataset
export MapDataset

export DataBunch
export train
export valid

export DummyCallback
export ProgressCallback
export Recorder

export Learner
export model
export data_bunch
export loss
export loss!
export opt
export opt!
export fit!
export add_cb!

export AvgMetric
export AvgLoss
export AvgSmoothLoss
export reset
export accumulate
export value
export name

include("dataset.jl")
include("databunch.jl")
include("learner.jl")
include("callback.jl")

include("metric.jl")
include("recorder.jl")
include("exercise.jl")

end
