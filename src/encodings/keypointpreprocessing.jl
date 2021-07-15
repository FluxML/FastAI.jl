struct KeypointTensor{N, T, M} <: Block
    sz::NTuple{M, Int}
end


mockblock(block::KeypointTensor{N}) where N = rand(SVector{N, Float32}, block.sz)
function checkblock(block::KeypointTensor{N, T}, data::AbstractArray{T}) where {N, T}
    return length(data) == (prod(block.sz) * N)
end

"""
    KeypointPreprocessing(bounds) <: Encoding

Scale a `Keypoints` block falling in a rectangle of `bounds` so
that they lie between -1 and 1.
"""
struct KeypointPreprocessing{N, T<:Number} <: Encoding
    bounds::NTuple{N, Int}
end
KeypointPreprocessing(bounds::NTuple{N, Int}) where N = KeypointPreprocessing{N, Float32}(bounds)

function encode(enc::KeypointPreprocessing{N, T}, context, block::Keypoints{N}, data) where {N, T}
    ks = map(k -> (SVector{N, T}(k) .* (convert(T, 2) ./ enc.bounds)) .- one(T), data)
    return reinterpret(T, ks)
end

function decode(enc::KeypointPreprocessing{N, T}, context, block::KeypointTensor{N}, data) where {N, T}
    ks = reshape(reinterpret(SVector{N, T}, data), block.sz)
    return map(k -> ((k) .+ one(T)) ./ (convert(T, 2) ./ SVector{N, T}(enc.bounds)), ks)
end

encodedblock(::KeypointPreprocessing{N, T}, block::Keypoints{N, M}) where {N, T, M} = KeypointTensor{N, T, M}(block.sz)
decodedblock(::KeypointPreprocessing{N}, block::KeypointTensor{N}) where N = Keypoints{N}(block.sz)
