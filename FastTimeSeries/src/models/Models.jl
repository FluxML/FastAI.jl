module Models

using ..FastAI

using Flux
using Zygote
using DataDeps
using InlineTest

# include("StackedLSTM.jl")
include("layers.jl")
include("RNN.jl")
include("InceptionTime.jl")

export StackedLSTM, RNNModel, InceptionTime

end