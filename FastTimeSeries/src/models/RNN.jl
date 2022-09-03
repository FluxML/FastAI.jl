tabular2rnn(X::AbstractArray{<:AbstractFloat, 3}) = permutedims(X, (1, 3, 2))

struct RNNModel{A, B}
    recbackbone::A
    finalclassifier::B
end

"""
    RNNModel(recbackbonem, outsize, recout[; kwargs...])

Creates a RNN model from the recurrent 'recbackbone' architecture. The output from this backbone
is passed through a dropout layer before a 'finalclassifier' block.

## Keyword arguments.

- `outsize`: The output size of the final classifier block. For single classification tasks,
    this would be the number of classes.
- `recout`: The output size of the `recbackbone` architecture.
- `dropout_rate`: Dropout probability for the dropout layer.
"""

function RNNModel(recbackbone;
                outsize,
                recout,
                kwargs...)
    return RNNModel(recbackbone, Dense(recout, outsize))
end

function (m::RNNModel)(X)
    X = tabular2rnn(X)
    Flux.reset!(m.recbackbone)
    # ChainRulesCore.ignore_derivatives() do
    #     Flux.reset!(m.recbackbone)
    # end
    X = m.recbackbone(X)[:, :, end]
    return m.finalclassifier(X)
end

Flux.@functor RNNModel 