"""
    GAP1d(output_size)

Create a Global Adaptive Pooling + Flatten layer.
"""
function GAP1d(output_size::Int)
    gap = AdaptiveMeanPool((output_size,))
    Chain(gap, Flux.flatten)    
end