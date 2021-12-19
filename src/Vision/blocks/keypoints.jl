
"""
    Keypoints{N}(sz) <: Block

A block representing an array of size `sz` filled with keypoints of type
`SVector{N}`.
"""
struct Keypoints{N,M} <: Block
    sz::NTuple{M,Int}
end
Keypoints{N}(n::Int) where {N} = Keypoints{N,1}((n,))
Keypoints{N}(t::NTuple{M,Int}) where {N,M} = Keypoints{N,M}(t)

function checkblock(
    ::Keypoints{N,M},
    ::AbstractArray{<:Union{SVector{N,T},Nothing},M},
) where {M,N,T}
    return true
end

mockblock(block::Keypoints{N}) where {N} = rand(SVector{N,Float32}, block.sz)


# ## Visualization

function showblock!(io, ::ShowText, block::Keypoints{2}, obs)
    print(io, UnicodePlots.scatterplot(first.(obs), last.(obs), marker=:cross))
end


function showblock!(io, ::ShowText, block::Bounded{2, <:Keypoints{2}}, obs)
    h, w = block.size
    plot = UnicodePlots.scatterplot(
        first.(obs), last.(obs),
        xlim=(0, w), ylim=(0, h), marker=:cross)
    print(io, plot)
end
