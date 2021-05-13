
"""
    ImageSegmentation(classes[, sz; kwargs...]) <: LearningMethod

A learning method for semantic image segmentation:
given an image and a set of classes, determine *for every pixel* which
class it falls into. For example, assign some pixels the class "road" and
others the class "background".

Images are resized and cropped to `sz` (see [`ProjectiveTransforms`](#))
and preprocessed using [`ImagePreprocessing`](#).
`classes` is a vector of the class labels.

## Keyword arguments

- `aug_projection::`[`DataAugmentation.Transform`](#)` = Identity()`: Projective
    augmentation to apply during training. See
    [`ProjectiveTransforms`](#) and [`augs_projection`](#).
- `aug_image::`[`DataAugmentation.Transform`](#)` = Identity()`: Other image
    augmentation to apply to cropped image during training. See
    [`ImagePreprocessing`](#) and [`augs_lighting`](#).
- `downscale::Int = 0`: Downscale the masks by a factor of `2^downscale`, i.e. for
    a value of `0`, the images and masks will have the same size and for a value of
    `1`, the target masks will have half the size of the input images.
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
mutable struct ImageSegmentation{N} <: DLPipelines.LearningMethod{ImageSegmentationTask}
    classes::AbstractVector
    downscale::Int
    projections::ProjectiveTransforms{N}
    imagepreprocessing::ImagePreprocessing
end


function Base.show(io::IO, method::ImageSegmentation)
    show(io, ShowTypeOf(method))
    fields = (
        classes = ShowLimit(ShowList(method.classes, brackets="[]"), limit=50),
        downscale = method.downscale,
        projections = method.projections,
        imageprepocessing = method.imagepreprocessing
    )
    show(io, ShowProps(fields, new_lines=true))
end


function ImageSegmentation(
        classes::AbstractVector,
        sz=(224, 224);
        aug_projection=Identity(),
        aug_image=Identity(),
        downscale=0,
        buffered=true,
        means=IMAGENET_MEANS,
        stds=IMAGENET_STDS,
        C=RGB{N0f8},
        T=Float32)

    projections = ProjectiveTransforms(sz;
        augmentations=aug_projection, buffered = buffered)
    imagepreprocessing = ImagePreprocessing(;
        means=means, stds=stds, C=C, T=T, augmentations=aug_image, buffered=buffered)
    return ImageSegmentation(
        classes, downscale,
        projections, imagepreprocessing)
end

# ## Core interface

DLPipelines.encode(method::ImageSegmentation, context, sample::NamedTuple) =
    DLPipelines.encode(method, context, Tuple(sample))

function DLPipelines.encode(
        method::ImageSegmentation,
        context,
        sample::Tuple)
    image, mask = sample
    imagec, maskc = run(
        method.projections,
        context,
        (DataAugmentation.Image(image), MaskMulti(mask, 1:length(method.classes))))

    x = run(method.imagepreprocessing, context, imagec)

    f = method.downscale
    if f != 0
        newsz = ntuple(i -> round(Int, size(image, i) * 1 / 2^f), ndims(image))
        ytfm = ScaleFixed(newsz) |> DataAugmentation.Crop(newsz, DataAugmentation.FromOrigin()) |> OneHot()
    else
        ytfm = OneHot()
    end
    y = apply(ytfm, MaskMulti(maskc, 1:length(method.classes))) |> itemdata
    return (x, y)
end


function DLPipelines.decodeŷ(method::ImageSegmentation, context, ŷ)
    return onecoldmask(ŷ)
end


function DLPipelines.mocksample(method::ImageSegmentation)
    inputsz = rand.(UnitRange.(method.projections.sz, method.projections.sz .* 2))
    return (
        input = rand(RGB{N0f8}, inputsz),
        target = rand(1:length(method.classes), inputsz)
    )
end


function DLPipelines.mockmodel(method::ImageSegmentation)
    return function segmodel(xs)
        outsz = (
            round.(Int, size(xs)[1:end-2] ./ 2^method.downscale)...,
            length(method.classes),
            size(xs)[end])
        return rand(Float32, outsz)
    end
end


onecoldmask(mask) = reshape(map(I -> I.I[end], argmax(mask; dims=ndims(mask))), size(mask)[1:end-1])


# ## Plotting interface

function plotsample!(f, method::ImageSegmentation, sample)
    image, mask = sample
    f[1, 1] = ax1 = imageaxis(f)
    f[1, 2] = ax2 = imageaxis(f)
    plotimage!(ax1, image)
    plotmask!(ax2, mask, method.classes, )
end


function plotxy!(f, method::ImageSegmentation, x, y)
    image = invert(method.imagepreprocessing, x)
    mask = decodeŷ(method, Inference(), y)
    f[1, 1] = ax1 = imageaxis(f)
    f[2, 1] = ax2 = imageaxis(f)
    plotimage!(ax1, image)
    plotmask!(ax2, mask, method.classes)
    return f
end

function plotprediction!(f, method::ImageSegmentation, x, ŷ, y)
    image = invert(method.imagepreprocessing, x)
    maskgt = decodeŷ(method, Inference(), y)
    maskpred = decodeŷ(method, Inference(), ŷ)
    f[1, 1] = ax1 = imageaxis(f, title="Image")
    f[2, 1] = ax2 = imageaxis(f, title="Pred")
    f[3, 1] = ax3 = imageaxis(f, title="GT")
    plotimage!(ax1, image)
    plotmask!(ax2, maskpred, method.classes)
    plotmask!(ax3, maskgt, method.classes)
    return f
end

# ## Training interface

function DLPipelines.methodmodel(method, backbone; kwargs...)
    return UNetDynamic(
        backbone,
        (method.projections.sz..., 3, 1),
        length(method.classes);
        fdownscale = method.downscale,
        kwargs...)

end


DLPipelines.methodlossfn(::ImageSegmentation) = segmentationloss


function segmentationloss(ypreds, ys; kwargs...)
    # Has to be reshaped to 3D array since `logitcrossentropy(...; dims = 3)` doesn't work on GPU
    ypreds = reshape(ypreds, :, size(ypreds, 3), size(ypreds, 4))
    ys = reshape(ys, :, size(ys, 3), size(ys, 4))
    Flux.Losses.logitcrossentropy(ypreds, ys; dims = 2, kwargs...)
end
