

"""
    ImageClassification(classes, sz[; augmentations, ...]) <: Method{ImageClassificationTask}
    ImageClassification(n, ...)

A [`LearningMethod`](#) for multi-class image classification using softmax probabilities.

`classes` is a vector of the category labels. Alternatively, you can pass an integer.
Images are resized to `sz`.

During training, a random crop is used and `augmentations`, a `DataAugmentation.Transform`
are applied.

### Types

- `input::AbstractMatrix{2, <:Colorant}`: an image
- `target` the class label that the image belongs to
- `x::AbstractArray{Float32, 3}`: a normalized 3D-array with dimensions *height, width, channels*
- `y::AbstractVector{Float32}`: one-hot encoding of category

### Model

- input size: `(sz..., ch, batch)` where `ch` depends on color type `C`.
- output size: `(nclasses, batch)`
"""
mutable struct ImageClassification <: DLPipelines.LearningMethod{ImageClassificationTask}
    sz::Tuple{Int, Int}
    classes::AbstractVector
    projectivetransforms::ProjectiveTransforms
    imagepreprocessing::ImagePreprocessing
end

Base.show(io::IO, method::ImageClassification) = print(
    io, "ImageClassification() with $(length(method.classes)) classes")

function ImageClassification(
        classes::AbstractVector,
        sz = (224, 224);
        augmentations = Identity(),
        means = IMAGENET_MEANS,
        stds = IMAGENET_STDS,
        C = RGB{N0f8},
        T = Float32
    )
    projectivetransforms = ProjectiveTransforms(sz, augmentations = augmentations)
    imagepreprocessing = ImagePreprocessing(means, stds; C = C, T = T)
    ImageClassification(sz, classes, projectivetransforms, imagepreprocessing)
end

ImageClassification(n::Int, args...; kwargs...) = ImageClassification(1:n, args...; kwargs...)


# Core interface implementation

function DLPipelines.encodeinput(
        method::ImageClassification,
        context,
        image)
    imagecropped = run(method.projectivetransforms, context, image)
    x = run(method.imagepreprocessing, context, imagecropped)
    return x
end


function DLPipelines.encodetarget(
        method::ImageClassification,
        context,
        category)
    idx = findfirst(isequal(category), method.classes)
    isnothing(idx) && error("`category` could not be found in `method.classes`.")
    return DataAugmentation.onehot(idx, length(method.classes))
end


function DLPipelines.encodetarget!(
        y::AbstractVector{T},
        method::ImageClassification,
        context,
        category) where T
    fill!(y, zero(T))
    idx = findfirst(isequal(category), method.classes)
    y[idx] = one(T)
    return y
end

#DLPipelines.encode(method::ImageClassification, context, (input, category)) = (DLPipelines.encodeinput(method, context, input), DLPipelines.encodetarget(method, context, category))
DLPipelines.encode(method::ImageClassification, context, inputCategoryTuple::NamedTuple) = DLPipelines.encode(method::ImageClassification, context, values(inputCategoryTuple))
DLPipelines.decodeŷ(method::ImageClassification, context, ŷ) = method.classes[argmax(ŷ)]

# Interpretation interface

DLPipelines.interpretinput(::ImageClassification, image) = image

function DLPipelines.interpretx(method::ImageClassification, x)
    return invert(method.imagepreprocessing, x)
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
            Dense(ch, length(method.classes)),
        )
    )
end

DLPipelines.methodlossfn(::ImageClassification) = Flux.Losses.logitcrossentropy

# Testing interface

function DLPipelines.mockinput(method::ImageClassification)
    inputsz = rand.(UnitRange.(method.sz, method.sz .* 2))
    return rand(RGB{N0f8}, inputsz)
end


function DLPipelines.mocktarget(method::ImageClassification)
    rand(1:length(method.classes))
end


function DLPipelines.mockmodel(method::ImageClassification)
    return xs -> rand(Float32, length(method.classes), size(xs)[end])
end
