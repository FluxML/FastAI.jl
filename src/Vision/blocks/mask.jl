
"""
    Mask{N, T}(classes) <: Block

Block for an N-dimensional categorical mask. `obs` is valid for
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

Base.nameof(::Mask{N}) where N = "Mask{$N}"

function FastAI.invariant_checkblock(block::Mask{N}; blockvar = "block", obsvar = "obs") where N
    return invariant([
            invariant("`$obsvar` is an `AbstractArray`",
                description = md("`$obsvar` should be of type `AbstractArray`.")) do obs
                if !(obs isa AbstractArray)
                    return "Instead, got invalid type `$(nameof(typeof(obs)))`." |> md
                end
            end,
            invariant("`$obsvar` is `$N`-dimensional") do obs
                if ndims(obs) != N
                    return "Instead, got invalid dimensionality `$N`." |> md
                end
            end,
            invariant("All elements are valid labels") do obs
                valid = ∈(block.classes).(obs)
                if !(all(valid))
                    unknown = unique(obs[valid .== false])
                    return md("""`$obsvar` should contain only valid labels,
                    i.e. `∀ y ∈ $obsvar: y ∈ $blockvar.classes`, but `$obsvar` includes
                    unknown labels: `$(sprint(show, unknown))`.

                    Valid classes are:
                    `$(sprint(show, block.classes, context=:limit => true))`""")
                end
            end,
        ],
        FastAI.__inv_checkblock_title(block, blockvar, obsvar),
        :seq
    )
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

_maskimage(mask::AbstractArray{<:Gray{T}}, args...) where {T} =
    _maskimage(reinterpret(T, mask), args...)
_maskimage(mask::AbstractArray{<:Normed{T}}, args...) where {T} =
    _maskimage(reinterpret(T, mask), args...)
