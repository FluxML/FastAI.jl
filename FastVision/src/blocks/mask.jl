
"""
    Mask{N, T}(classes) <: Block

Block for an N-dimensional categorical mask. `obs` is valid for
`Mask{N, T}(classes)`
if it is an N-dimensional array with every element in `classes`.
"""
struct Mask{N, T} <: Block
    classes::AbstractVector{T}
end
Mask{N}(classes::AbstractVector{T}) where {N, T} = Mask{N, T}(classes)

function checkblock(block::Mask{N, T}, a::AbstractArray{T, N}) where {N, T}
    return all(map(x -> x ∈ block.classes, a))
end

function mockblock(mask::Mask{N, T}) where {N, T}
    rand(mask.classes, ntuple(_ -> 16, N))::AbstractArray{T, N}
end

# Visualization

function showblock!(io, ::ShowText, block::Mask{2}, obs)
    img = _maskimage(obs, block.classes)
    ImageInTerminal.imshow(io, img)
end

function _maskimage(mask, classes)
    classtoidx = Dict(class => i for (i, class) in enumerate(classes))
    colors = distinguishable_colors(length(classes), transform = deuteranopic)
    return map(x -> colors[classtoidx[x]], mask)
end

function _maskimage(mask::AbstractArray{<:Gray{T}}, args...) where {T}
    _maskimage(reinterpret(T, mask), args...)
end
function _maskimage(mask::AbstractArray{<:Normed{T}}, args...) where {T}
    _maskimage(reinterpret(T, mask), args...)
end

@testset "OneHot [encoding]" begin
    enc = OneHot()
    testencoding(enc, Mask{2}(1:10), rand(1:10, 50, 50))
end
