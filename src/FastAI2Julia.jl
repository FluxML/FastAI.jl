#=
FastAI2Julia.jl:

Author: Peter Wolf (opus111@gmail.com)

A first cut at a port of the FastAI V2 API to Julia

The original source is here

https://github.com/fastai/fastai2/blob/master/fastai2/

The documentation is copied from here

https://github.com/fastai/fastai2/blob/master/docs/

The main purpose of this code is to see if the team likes the method
of defining an interface and implementations in Julia
=#

module FastAI2Julia

export fit
export add_cb
export AbstractCallback
export TrainEvalCallback
export Learner


include("dataset.jl")
include("learner.jl")
include("callback.jl")

end
