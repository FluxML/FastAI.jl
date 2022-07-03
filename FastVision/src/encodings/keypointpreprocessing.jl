"""
    KeypointTensor{N, T, M} <: Block

Block for encoded [`Keypoints`](#)`{N, T, M}` returned by
[`KeypointPreprocessing`](#).
"""
struct KeypointTensor{N, T, M} <: Block
    sz::NTuple{M, Int}
end

mockblock(block::KeypointTensor{N}) where {N} = rand(SVector{N, Float32}, block.sz)
function checkblock(block::KeypointTensor{N, T}, obs::AbstractArray{T}) where {N, T}
    return length(obs) == (prod(block.sz) * N)
end

"""
    KeypointPreprocessing(bounds) <: Encoding

Scale a `Keypoints` block falling in a rectangle of `bounds` so
that they lie between -1 and 1.
"""
struct KeypointPreprocessing{N, T <: Number} <: Encoding
    bounds::NTuple{N, Int}
end
function KeypointPreprocessing(bounds::NTuple{N, Int}) where {N}
    KeypointPreprocessing{N, Float32}(bounds)
end

function encode(enc::KeypointPreprocessing{N, T}, context, block::Keypoints{N},
                obs) where {N, T}
    ks = map(k -> (SVector{N, T}(k) .* (convert(T, 2) ./ enc.bounds)) .- one(T), obs)
    return reinterpret(T, ks)
end

function decode(enc::KeypointPreprocessing{N, T}, context, block::KeypointTensor{N},
                obs) where {N, T}
    ks = reshape(reinterpret(SVector{N, T}, obs), block.sz)
    return map(k -> ((k) .+ one(T)) ./ (convert(T, 2) ./ SVector{N, T}(enc.bounds)), ks)
end

function encodedblock(::KeypointPreprocessing{N, T}, block::Keypoints{N, M}) where {N, T, M}
    KeypointTensor{N, T, M}(block.sz)
end
function decodedblock(::KeypointPreprocessing{N}, block::KeypointTensor{N}) where {N}
    Keypoints{N}(block.sz)
end

# ## Optional interfaces

# The default loss function to compare encoded keypoints is Mean Squared Error:

function blocklossfn(outblock::KeypointTensor{N}, yblock::KeypointTensor{N}) where {N}
    outblock.sz == yblock.sz || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end

# ## Tests

@testset "KeypointPreprocessing [encoding]" begin
    ks = [
        SVector{2, Float32}(10, 10),
        SVector{2, Float32}(50, 80),
    ]
    sz = (100, 100)

    block = Keypoints{2}(2)
    enc = KeypointPreprocessing(sz)
    ctx = Training()

    testencoding(enc, block, ks)
    y = encode(enc, ctx, block, ks)
    ks_ = decode(enc, ctx, encodedblock(enc, block), y)
    @test ks ≈ ks_
end
