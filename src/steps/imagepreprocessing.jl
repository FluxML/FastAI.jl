# See **[`ImagePreprocessing`](#).**

"""
    ImagePreprocessing(means, stds[; augmentations, C = RGB{N0f8}, T = Float32])

Converts an image to a color `C`, then to a 3D-array of type `T` and
finally normalizes the values using `means` and `stds`.

If no `means` or `stds` are given, uses ImageNet statistics.
"""
struct ImagePreprocessing
    # hold one copy of the transform for every context
    # in case the input sizes differ based on `context`
    # since that would create problems with the buffers
    traintfm
    validtfm
    inferencetfm
end

function ImagePreprocessing(
        means::SVector{N} = IMAGENET_MEANS,
        stds::SVector{N} = IMAGENET_STDS;
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
    return DataAugmentation.apply(tfm, DataAugmentation.Image(image)) |> itemdata
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
