module Models

using ..FastAI

using Flux
using Zygote
using DataDeps
using InlineTest

include("StackedLSTM.jl")
include("RNN.jl")

export StackedLSTM, RNNModel

end