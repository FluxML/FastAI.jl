abstract type SingleKeypointRegressionTask <: DLPipelines.LearningTask end

"""
    SingleKeypointRegression(classes[, sz; kwargs...]) <: LearningMethod

A learning method for single keypoint regression: given an image, find the location (in
pixels) of a keypoint. For example, find the center of a person's head.

Images are resized and cropped to `sz` (see [`ProjectiveTransforms`](#)) and preprocessed
using [`ImagePreprocessing`](#).

## Keyword arguments

- `aug_projection::`[`DataAugmentation.Transform`](#)` = Identity()`: Projective
    augmentation to apply during training. See
    [`ProjectiveTransforms`](#) and [`augs_projection`](#).
- `aug_image::`[`DataAugmentation.Transform`](#)` = Identity()`: Other image
    augmentation to apply to cropped image during training. See
    [`ImagePreprocessing`](#) and [`augs_lighting`](#).
- `buffered = true`: Whether to use inplace transformations when projecting and
  preprocessing image. Reduces memory usage.
- `means = IMAGENET_MEANS` and `stds = IMAGENET_STDS`: Color channel means and
    standard deviations to use for normalizing the image.

## Learning method reference

This learning method implements the following interfaces:

{.tight}
- Core interface
- Plotting interface
- Training interface
- Testing interface

### Types

Types of data throughout the DLPipelines.jl pipeline.

- **`sample`**: `Tuple`/`NamedTuple` of
    - **`input`**`::AbstractArray{2, T}`: A 2-dimensional array with dimensions (height, width)
        and elements of a color or number type. `Matrix{RGB{Float32}}` is a 2D RGB image,
        while `Array{Float32, 3}` would be a 3D grayscale image. If element type is a number
        it should fall between `0` and `1`. It is recommended to use the `Gray` color type
        to represent grayscale images.
    - **`target`**`::AbstractArray{2, <:Integer}`: A mask with the integer at each pixel giving
        a class index into `classes`.
- **`x`**`::AbstractArray{Float32, 3}`: a normalized array with dimensions
    `(height, width, color channels)`. See [`ImagePreprocessing`](#) for additional information.
- **`y`**`::AbstractArray{Float32, 3}`: a one-hot encoded array with dimensions
    `(height/2^downscale, width/2^downscale, length(classes))` with the class index of the
    corresponding pixel set to `1.` and other values to `0.`.
- **`ŷ`**`::AbstractVector{Float32}`: vector of predicted class scores.

### Model sizes

Array sizes that compatible models must conform to.

- Full model: `(sz..., 3, batch) -> ((sz ./ 2^downscale)..., batch)`
- Backbone model: `(sz..., 3, batch) -> ((sz ./ f)..., ch, batch)` where `f`
    is a downscaling factor `f = 2^k`. `methodmodel` will build a U-Net model
    from the backbone by inserting an upscaling layer and skip connections for
    every downscaling layer in the original network.

It is recommended *not* to use [`Flux.softmax`](#) as the final layer for custom models,
as for numerical stability, the loss function takes in the logits.
"""
struct SingleKeypointRegression{N} <: DLPipelines.LearningMethod{SingleKeypointRegressionTask}
    projections::ProjectiveTransforms{N}
    imagepreprocessing::ImagePreprocessing
end

function SingleKeypointRegression(
        sz=(224, 224);
        aug_projection=Identity(),
        aug_image=Identity(),
        buffered=true,
        means=IMAGENET_MEANS,
        stds=IMAGENET_STDS,
        C=RGB{N0f8},
        T=Float32)

    projections = ProjectiveTransforms(sz;
        augmentations=aug_projection, buffered = buffered)
    imagepreprocessing = ImagePreprocessing(;
        means=means, stds=stds, C=C, T=T, augmentations=aug_image, buffered=buffered)
    return SingleKeypointRegression(projections, imagepreprocessing)
end

# Core interface

function DLPipelines.encode(method::SingleKeypointRegression, context, sample::Union{Tuple, NamedTuple})
    image, keypoint = sample[1], sample[2]
    pimage, pkeypoints = FastAI.run(method.projections, context, (image, [keypoint]))
    x = FastAI.run(method.imagepreprocessing, context, pimage)
    y = collect(Float32.(scalepoint(pkeypoints[1], method.projections.sz)))
    return x, y
end

function DLPipelines.decodeŷ(method::SingleKeypointRegression, context, ŷ)
    return ((ŷ ) .+ 1) ./ (2 ./ method.projections.sz)
end

scalepoint(v, sz) = v .* (2 ./ sz) .- 1

# Plotting interface

function FastAI.plotsample!(f, method::SingleKeypointRegression, sample)
    image, v = sample
    f[1, 1] = ax1 = imageaxis(f)
    _drawkeypoint!(image, v)
    plotimage!(ax1, image)
end


function FastAI.plotxy!(f, method::SingleKeypointRegression, x, y)
    image = FastAI.invert(method.imagepreprocessing, x)
    v = decodeŷ(method, Validation(),  y)
    _drawkeypoint!(image, v)

    ax1 = f[1, 1] = FastAI.imageaxis(f)
    plotimage!(ax1, image)

    return f
end


function FastAI.plotprediction!(f, method::SingleKeypointRegression, x, ŷ, y)
    image = FastAI.invert(method.imagepreprocessing, x)
    v = decodeŷ(method, Validation(), y)
    v̂ = decodeŷ(method, Validation(), ŷ)

    _drawkeypoint!(image, v, c = RGB(0, 1, 0))
    _drawkeypoint!(image, v̂, c = RGB(1, 0, 0))
    ax1 = f[1, 1] = FastAI.imageaxis(f)
    plotimage!(ax1, image)

    return f
end


_toindex(v) = CartesianIndex(Tuple(round.(Int, v)))
_boxIs(I, r) = I-(r*CartesianIndex(1, 1)):I+(r*CartesianIndex(1, 1))
function _drawkeypoint!(img, v; r = 2, c = RGB(0, 0, 1))
    Is = _boxIs(_toindex(v), r)
    for I in Is
        checkbounds(Bool, img, I) && (img[I] = c)
    end
    return img
end


# Training interface

DLPipelines.methodlossfn(method::SingleKeypointRegression) = Flux.mse

function DLPipelines.methodmodel(method::SingleKeypointRegression, backbone)
    h, w, ch, b = Flux.outdims(backbone, (method.projections.sz..., 3, 1))
    head = FastAI.Models.visionhead(ch, 2, y_range=(-1, 1))
    return Chain(backbone, head)
end


# Testing interface

function DLPipelines.mockinput(method::SingleKeypointRegression)
    inputsz = rand.(UnitRange.(method.projections.sz, method.projections.sz .* 2))
    return rand(RGB{N0f8}, inputsz)
end


function DLPipelines.mocktarget(method::SingleKeypointRegression)
    sz = method.projections.sz
    return SVector{2, Float32}(rand(1:sz[1]), rand(1:sz[2]))
end


function DLPipelines.mockmodel(::SingleKeypointRegression)
    return xs -> rand(Float32, 2, size(xs)[end])
end
