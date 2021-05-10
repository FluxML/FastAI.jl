# See **[`ImagePreprocessing`](#).**

"""
    ImagePreprocessing([; kwargs...])

A helper for building learning methods that need to preprocess images.
Preprocessing consists of applying color `augmentations` like [`augs_lighting`]
(only during training), conversion to a common color type `C`, expanding color
channels to an array dimension and normalizing the values.

Apply to an image using [`FastAI.run`](#) or inplace using [`FastAI.run!`](#).

The only difference between different contexts is that augmentations are only
applied during [`DLPipelines.Training`](#).

## Keyword arguments

- `augmentations::`[`DataAugmentation.Transform`](#): Augmentation to apply to every image
    before preprocessing. See [`augs_lighting`](#)
- `buffered = true`: Whether to use inplace transformations. Reduces memory usage.
- `means::SVector = IMAGENET_MEANS`: mean value of each color channel.
- `stds::SVector = IMAGENET_STDS`: standard deviation of each color channel.
- `C::Type{<:Colorant} = RGB{N0f8}`: color type to convert images to.
- `T::Type{<:Real} = Float32`: element type of output

## Examples


{cell=ImagePreprocessing, output=false}
```julia
using FastAI, TestImages
img = testimage("lighthouse")
preprocessing = FastAI.ImagePreprocessing(augmentations=augs_lighting())
a = FastAI.run(preprocessing, Training(), img)
summary(a)
```

"""
struct ImagePreprocessing
    traintfm
    validtfm
    inferencetfm
end

function ImagePreprocessing(;
        means::SVector{N} = IMAGENET_MEANS,
        stds::SVector{N} = IMAGENET_STDS,
        augmentations = Identity(),
        C = RGB{N0f8},
        T = Float32,
        buffered = true) where N
    # TODO: tensor of type T
    tfms = (
        augmentations |> ToEltype(C) |> ImageToTensor() |> Normalize(means, stds),
        ToEltype(C) |> ImageToTensor() |> Normalize(means, stds),
        ToEltype(C) |> ImageToTensor() |> Normalize(means, stds),
    )
    if buffered
        tfms = (
            BufferedThreadsafe(tfms[1]),
            BufferedThreadsafe(tfms[2]),
            # Inference transform is not buffered since it can have
            # varying sizes
            tfms[3]
        )
    end

    return ImagePreprocessing(tfms...)
end


function ImagePreprocessing(means::NTuple{N}, stds::NTuple{N}; kwargs...) where N
    return ImagePreprocessing(SVector{N}(means), SVector{N}(stds); kwargs...)
end

Base.show(io::IO, ::ImagePreprocessing) = print(io, "ImagePreprocessing()")


function run(ip::ImagePreprocessing, context::Context, image)
    tfm = _gettfm(ip, context)
    x = DataAugmentation.apply(tfm, DataAugmentation.Image(image)) |> itemdata
    return deepcopy(x)
end

function run!(x, ip::ImagePreprocessing, context::Context, image)
    tfm = _gettfm(ip, context)
    DataAugmentation.apply!(ArrayItem(x), tfm, DataAugmentation.Image(image)) |> itemdata
    return x
end


_gettfm(ip::ImagePreprocessing, context::Training) = ip.traintfm
_gettfm(ip::ImagePreprocessing, context::Validation) = ip.validtfm
_gettfm(ip::ImagePreprocessing, context::Inference) = ip.inferencetfm



function invert(ip::ImagePreprocessing, x::AbstractArray{T, N}) where {N, T}
    tfm = _parenttfm(ip.traintfm).transforms[end]
    return DataAugmentation.tensortoimage(DataAugmentation.denormalize(x, tfm.means, tfm.stds))
end


const IMAGENET_MEANS = SVector{3, Float32}(.485, 0.456, 0.406)
const IMAGENET_STDS = SVector{3, Float32}(0.229, 0.224, 0.225)
