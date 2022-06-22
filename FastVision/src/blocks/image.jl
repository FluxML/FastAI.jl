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
using FastAI, FastVision
FastAI.mockblock(Image{2}())
```

To visualize a 2D-image observation, use [`showblock`](#). This is supported for
both the `ShowText` and the `ShowMakie` backend.

```julia
showblock(Image{2}(), rand(RGB{N0f8}, 10, 10))
```


"""
struct Image{N} <: Block end

checkblock(::Image{N}, ::AbstractArray{T, N}) where {T <: Union{Colorant, Number}, N} = true
mockblock(::Image{N}) where {N} = rand(RGB{N0f8}, ntuple(_ -> 16, N))

setup(::Type{Image}, data) = Image{ndims(getobs(data, 1))}()

# Visualization

function showblock!(io, ::ShowText, block::Image{2}, obs)
    ImageInTerminal.imshow(io, obs)
end
