"""
    Image{N}() <: Block

[`Block`](#) for an N-dimensional image. `obs` is valid for `Image{N}()`
if it is an N-dimensional array with color or number element type.

## Examples

Creating a block:

```julia
Image{2}()  # 2D-image
Image{3}()  # 3D-image
```

Example valid images:

```julia
@test checkblock(Image{2}(), rand(RGB, 10, 10))         # Color image
@test checkblock(Image{2}(), rand(10, 10))              # Numbers treated as grayscale
@test checkblock(Image{3}(), rand(Gray{N0f8}, 10, 10, 10))  # Grayscale 3D-image
```

The color channels (if any) are not counted as a dimension and represented
through color types like `RGB{N0f8}`:

```julia
@test !checkblock(Image{2}, rand(10, 10, 3))  # Not a 2D image
```

You can create a random observation using [`mockblock`](#):

{cell=main}
```julia
using FastAI
FastAI.mockblock(Image{2}())
```

To visualize a 2D-image observation, use [`showblock`](#). This is supported for
both the `ShowText` and the `ShowMakie` backend.

```julia
showblock(Image{2}(), rand(RGB{N0f8}, 10, 10))
```


"""
struct Image{N} <: Block end

checkblock(::Image{N}, ::AbstractArray{T,N}) where {T<:Union{Colorant,Number},N} = true
mockblock(::Image{N}) where {N} = rand(RGB{N0f8}, ntuple(_ -> 16, N))

setup(::Type{Image}, data) = Image{ndims(getobs(data, 1))}()

Base.nameof(::Image{N}) where N = "Image{$N}"

# Visualization

showblock!(io, ::ShowText, block::Image{2}, obs::AbstractMatrix{<:Colorant}) =
    ImageInTerminal.imshow(io, obs)
showblock!(io, ::ShowText, block::Image{2}, obs::AbstractMatrix{<:Real}) =
    ImageInTerminal.imshow(io, colorview(Gray, obs))


function FastAI.invariant_checkblock(block::Image{N}; blockvar = "block", obsvar = "obs") where N
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
            invariant("`$obsvar` should have a color or numerical element type") do obs
                if !((eltype(obs) <: Color) ||(eltype(obs) <: Real))
                    return "Instead, got invalid element type `$(eltype(obs))`." |> md
                end
            end,
        ],
        FastAI.__inv_checkblock_title(block, blockvar, obsvar),
        :seq
    )
end

#=

function isblockinvariant(block::Image{N}; obsvar = "data", blockvar = "block") where {N}
    return SequenceInvariant(
        [
            BooleanInvariant(
                obs -> obs isa AbstractArray,
                name = "Image data is an array",
                messagefn = obs -> """Expected `$obsvar` to be a subtype of
                    `AbstractArray`, but instead got type `$(typeof(obs))`.""",
            ),
            BooleanInvariant(
                obs -> ndims(obs) == N,
                name = "Image data is `$N`-dimensional",
                messagefn = obs -> """Expected `$obsvar` to be an `$N`-dimensional array,
                    but instead got a `$(ndims(obs))`-dimensional array.""",
            ),
            BooleanInvariant(
                obs -> eltype(obs) <: Color || eltype(obs) <: Number,
                name = "Image data has a color or numerical type.",
                messagefn = obs -> """Expected `$obsvar` to have an element type that is a
                    color (`eltype($obsvar) <: Color`) or a number (`eltype($obsvar)
                    <: Color`), but instead found `eltype($obsvar) == $(eltype(obs)).`
                    """
            )
        ],
        "`$obsvar` is a valid `$(typeof(block))`",
        ""
    )
end
=#
