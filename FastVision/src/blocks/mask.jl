
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

mockblock(mask::Mask{N, T}) where {N, T} = rand(mask.classes, ntuple(_ -> 16, N))::AbstractArray{T, N}

function FastAI.invariant_checkblock(block::Mask{N}; blockvar = "block", obsvar = "obs", kwargs...) where N
    return invariant(
        FastAI.__inv_checkblock_title(block, blockvar, obsvar),
        [
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
        ];
        kwargs...
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
