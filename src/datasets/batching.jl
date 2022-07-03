
batchsize(batch::Union{Tuple, NamedTuple}) = batchsize(batch[1])
batchsize(batch::Dict) = batchsize(batch[first(keys(batch))])
batchsize(batch::AbstractArray{T, N}) where {T, N} = size(batch, N)

unbatch(batch) = collect(obsslices(batch))

obsslices(batch) = (obsslice(batch, i) for i in 1:batchsize(batch))

function obsslice(batch::AbstractArray{T, N}, i) where {T, N}
    return view(batch, [(:) for _ in 1:(N - 1)]..., i)
end

obsslice(batch::AbstractVector, i) = batch[i]

function obsslice(batch::Tuple, i)
    return Tuple(obsslice(batch[j], i) for j in 1:length(batch))
end

function obsslice(batch::NamedTuple, i)
    return (; zip(keys(batch), obsslice(values(batch), i))...)
end

function obsslice(batch::Dict, i)
    return Dict(k => obsslice(v, i) for (k, v) in batch)
end
