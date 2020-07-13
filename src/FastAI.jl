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

include("dataset.jl")

include("learner.jl")
include("callback.jl")

include("metric.jl")
include("recorder.jl")

export fit
export add_cb
export AbstractCallback
export TrainEvalCallback

export Learner
export current_batch
export batch_size
export loss

export AvgMetric
export AvgLoss
export AvgSmoothLoss
export reset
export accumulate
export value
export name



end
