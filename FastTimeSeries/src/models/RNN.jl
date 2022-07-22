function tabular2rnn(X::AbstractArray{Float32, 3})
    X = permutedims(X, (2, 1, 3))
    X = [X[t, :, :] for t ∈ 1:size(X, 1)]
    return X
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
    RNNModel(recbackbone, Dense(recout, outsize); kwargs...)
end

function RNNModel(recbackbone,
                  finalclassifier;
                  dropout_rate = 0.0)
    
    dropout = dropout_rate == 0 ? identity : Dropout(dropout_rate)
    Chain(tabular2rnn, recbackbone, dropout, finalclassifier)
end