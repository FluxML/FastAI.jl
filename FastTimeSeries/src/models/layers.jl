"""
    GAP1d(output_size)

Create a Global Adaptive Pooling + Flatten layer.
"""
function GAP1d(output_size::Int)
    gap = AdaptiveMeanPool((output_size,))
    Chain(gap, Flux.flatten)    
end

"""
	StackedLSTM(in, out, hiddensize, layers)

Stacked LSTM network. Feeds the data through a chain of LSTM layers, where the hidden state
of the previous layer gets fed to the next one. The first layer corresponds to
`LSTM(in, hiddensize)`, the hidden layers to `LSTM(hiddensize, hiddensize)`, and the final
layer to `LSTM(hiddensize, out)`. Takes the keyword argument `init` for the initialization
of the layers.

"""
function StackedLSTM(c_in::Int, c_out::Integer, hiddensize::Integer, layers::Integer;
			init=Flux.glorot_uniform)
	if layers == 1
		return Chain(LSTM(c_in, c_out; init=init))
	elseif layers == 2
		return Chain(LSTM(c_in, hiddensize; init=init),
					  LSTM(hiddensize, c_out; init=init))
	else
		chain_vec = [LSTM(c_in, hiddensize; init=init)]
		for i = 1:layers - 2
			push!(chain_vec, LSTM(hiddensize, hiddensize; init=init))
		end
		return Chain(chain_vec..., LSTM(hiddensize, c_out; init=init))
	end
end

function Conv1d(ni, nf, ks; stride = 1, padding = Flux.SamePad, dilation = 1, bias = true)
    return Conv(
        (ks,),
        ni => nf,
        stride = stride,
        pad = ks ÷ 2 * dilation;
        bias = false
    )
end