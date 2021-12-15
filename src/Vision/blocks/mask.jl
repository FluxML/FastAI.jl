
"""
    Mask{N, T}(classes) <: Block

Block for an N-dimensional categorical mask. `data` is valid for
`Mask{N, T}(classes)`
if it is an N-dimensional array with every element in `classes`.
"""
struct Mask{N,T} <: Block
    classes::AbstractVector{T}
end
Mask{N}(classes::AbstractVector{T}) where {N,T} = Mask{N,T}(classes)

function checkblock(block::Mask{N,T}, a::AbstractArray{T,N}) where {N,T}
    return all(map(x -> x ∈ block.classes, a))
end

mockblock(mask::Mask{N, T}) where {N, T} = rand(mask.classes, ntuple(_ -> 16, N))::AbstractArray{T, N}


# Visualization

function showblock!(io, ::ShowText, block::Mask{2}, data)
    img = _maskimage(data, block.classes)
    ImageInTerminal.imshow(io, img)
end


function _maskimage(mask, classes)
    classtoidx = Dict(class => i for (i, class) in enumerate(classes))
    colors = distinguishable_colors(length(classes), transform = deuteranopic)
    return map(x -> colors[classtoidx[x]], mask)
end

_maskimage(mask::AbstractArray{<:Gray{T}}, args...) where {T} =
    _maskimage(reinterpret(T, mask), args...)
_maskimage(mask::AbstractArray{<:Normed{T}}, args...) where {T} =
    _maskimage(reinterpret(T, mask), args...)
