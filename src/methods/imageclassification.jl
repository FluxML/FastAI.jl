
abstract type ImageClassificationTask <: DLPipelines.LearningTask end

"""
    ImageClassification(categories, sz[; augmentations, ...]) <: Method{ImageClassificationTask}
    ImageClassification(n, ...)

A [`Method`](#) for multi-class image classification using softmax probabilities.

`categories` is a vector of the category labels. Alternatively, you can pass an integer.
Images are resized to `sz`.

During training, a random crop is used and `augmentations`, a `DataAugmentation.Transform`
are applied.

### Types

- `input::AbstractMatrix{2, <:Colorant}`: an image
- `target::Int` the category that the image belongs to
- `x::AbstractArray{Float32, 3}`: a normalized 3D-array with dimensions *height, width, channels*
- `y::AbstractVector{Float32}`: one-hot encoding of category

### Model

- input size: `(sz..., ch, batch)` where `ch` depends on color type `C`.
- output size: `(nclasses, batch)`
"""
mutable struct ImageClassification <: DLPipelines.LearningMethod{ImageClassificationTask}
    sz::Tuple{Int, Int}
    categories::AbstractVector
    spatialtransforms::ProjectiveTransforms
    imagepreprocessing::ImagePreprocessing
end

Base.show(io::IO, method::ImageClassification) = print(
    io, "ImageClassification() with $(length(method.categories)) classes")

function ImageClassification(
        categories::AbstractVector,
        sz = (224, 224);
        augmentations = Identity(),
        means = IMAGENET_MEANS,
        stds = IMAGENET_STDS,
        C = RGB{N0f8},
        T = Float32
    )
    spatialtransforms = ProjectiveTransforms(sz, augmentations = augmentations)
    imagepreprocessing = ImagePreprocessing(means, stds; C = C, T = T)
    ImageClassification(sz, categories, spatialtransforms, imagepreprocessing)
end

ImageClassification(n::Int, args...; kwargs...) = ImageClassification(1:n, args...; kwargs...)


# Core interface implementation

function DLPipelines.encodeinput(
        method::ImageClassification,
        context,
        image)
    imagecropped = run(method.spatialtransforms, context, image)
    x = run(method.imagepreprocessing, context, imagecropped)
    return x
end


function DLPipelines.encodetarget(
        method::ImageClassification,
        context,
        category)
    idx = findfirst(isequal(category), method.categories)
    isnothing(idx) && error("`category` could not be found in `method.categories`.")
    return DataAugmentation.onehot(idx, length(method.categories))
end


function DLPipelines.encodetarget!(
        y::AbstractVector{T},
        method::ImageClassification,
        context,
        category) where T
    fill!(y, zero(T))
    idx = findfirst(isequal(category), method.categories)
    y[idx] = one(T)
    return y
end

DLPipelines.decodeŷ(method::ImageClassification, context, ŷ) = method.categories[argmax(ŷ)]

# Interpetration interface

DLPipelines.interpretinput(task::ImageClassification, image) = image

function DLPipelines.interpretx(task::ImageClassification, x)
    return invert(task.imagepreprocessing, x)
end


function DLPipelines.interprettarget(task::ImageClassification, class)
    return "Class $class"
end


# Training interface

function DLPipelines.methodmodel(method::ImageClassification, backbone)
    h, w, ch, b = Flux.outdims(backbone, (method.sz..., 3, 1))
    return Chain(
        backbone,
        Chain(
            AdaptiveMeanPool((1,1)),
            flatten,
            Dense(ch, length(method.categories)),
        )
    )
end

DLPipelines.methodlossfn(::ImageClassification) = Flux.Losses.logitcrossentropy

# Testing interface

function DLPipelines.mockinput(method)
    inputsz = rand.(UnitRange.(method.sz, method.sz .* 2))
    return rand(RGB{N0f8}, inputsz)
end


function DLPipelines.mocktarget(method)
    rand(1:length(method.categories))
end


function DLPipelines.mockmodel(method)
    return xs -> rand(Float32, length(method.categories), size(xs)[end])
end
