

const IMAGENET_MEANS = SVector{3,Float32}(.485, 0.456, 0.406)
const IMAGENET_STDS = SVector{3,Float32}(0.229, 0.224, 0.225)


"""
    ImageTensor{N} <: Block

Block for N+1-dimensional arrays representing an N-dimensional
image with the color channels expanded.

"""
struct ImageTensor{N} <: Block
    nchannels::Int
end

function checkblock(block::ImageTensor{N}, a::AbstractArray{T,M}) where {M,N,T}
    # Tensor has dimensionality one higher and color channels need to be the same
    return (N + 1 == M) && (size(a, M) == block.nchannels)
end


"""
    ImagePreprocessing([; kwargs...]) <: Encoding

Encodes `Image`s by converting them to a common color type `C`,
expanding the color channels and normalizing the channel values.
Additionally, apply pixel-level augmentations passed in as `augmentations`
during `Training`.

Currently works with 2D images only, but this constraint will be
removed in the future.

Encodes
- `Image{2}` -> `ImageTensor{3}`

## Keyword arguments

- `augmentations::`[`DataAugmentation.Transform`](#): Augmentation to apply to every image
    before preprocessing. See [`augs_lighting`](#)
- `buffered = true`: Whether to use inplace transformations. Reduces memory usage.
- `means::SVector = IMAGENET_MEANS`: mean value of each color channel.
- `stds::SVector = IMAGENET_STDS`: standard deviation of each color channel.
- `C::Type{<:Colorant} = RGB{N0f8}`: color type to convert images to.
- `T::Type{<:Real} = Float32`: element type of output

"""
struct ImagePreprocessing{P,N,C <: Color{P,N},T <: Number} <: Encoding
    buffered::Bool
    augmentations::DataAugmentation.Transform
    stats::Tuple{SVector{N},SVector{N}}
    tfms::Dict{Context,DataAugmentation.Transform}
end


function ImagePreprocessing(;
        means::SVector{N}=IMAGENET_MEANS,
        stds::SVector{N}=IMAGENET_STDS,
        augmentations=Identity(),
        C::Type{<:Color{U,N}}=RGB{N0f8},
        T=Float32,
        buffered=true) where {N,U}
    # TODO: tensor of type T
    stats = means, stds
    basetfm = ToEltype(C) |> ImageToTensor{T}() |> Normalize(means, stds)
    if buffered
        tfms = Dict(
            Training() => BufferedThreadsafe(augmentations |> basetfm),
            Validation() => BufferedThreadsafe(basetfm),
            # Inference transform is not buffered since it can have
            # varying sizes
            Inference() => basetfm,
        )
    else
        tfms = Dict(
            Training() => augmentations |> basetfm,
            Validation() => basetfm,
            Inference() => basetfm,
        )
    end

    return ImagePreprocessing{U,N,C,T}(buffered, augmentations, stats, tfms)
end

colorchannels(C::Type{<:Color{T,N}}) where {T,N} = N

function encodedblock(ip::ImagePreprocessing{P,M,C}, ::Image{N}) where {P,M,C,N}
    return ImageTensor{N}(colorchannels(C))
end

function encode(ip::ImagePreprocessing, context, block::Image, data)
    return copy(apply(ip.tfms[context], DataAugmentation.Image(data)) |> itemdata)
end

decodedblock(::ImagePreprocessing, ::ImageTensor{N}) where N = Image{N}()

function decode(ip::ImagePreprocessing, context, block::ImageTensor, data)
    means, stds = ip.stats
    return copy(DataAugmentation.tensortoimage(DataAugmentation.denormalize(data, means, stds)))
end

# Setup and image statistic calculation

"""
    imagestats(image, C)

Compute the color channel-wise means and standard deviations of all pixels.
`image` is converted to color type `C` (e.g. `RGB{N0f8}`, `Gray{N0f8}`)
before statistics are calculated.
"""
function imagestats(img::AbstractArray{T,N}, C) where {T,N}
    imt = DataAugmentation.imagetotensor(map(x -> convert(C, x), img))
    means = reshape(mean(imt; dims=1:N), :)
    stds = reshape(std(imt; dims=1:N), :)
    return means, stds
end


"""
    imagedatasetstats(data, C[; parallel, progress])

Given a data container of images `data`, compute the color channel-wise means
and standard deviations across all observations. Images are converted to color type
`C` (e.g. `RGB{N0f8}`, `Gray{N0f8}`) before statistics are calculated.

If `progress = true`, show a progress bar.
"""
function imagedatasetstats(
        data,
        C;
        progress=true,
        progressfn=progress ? tqdm : identity)
    means, stds = imagestats(getobs(data, 1), C)
    loaderfn = d -> eachobsparallel(d, buffered=false, useprimary=true)

    for (means_, stds_) in mapobs(img -> imagestats(img, C), data) |> loaderfn |> progressfn
        means .+= means_
        stds .+= stds_
    end
    return means ./ nobs(data), stds ./ nobs(data)
end


function setup(::Type{ImagePreprocessing}, ::Image, data; C = RGB{N0f8}, progress = false, kwargs...)
    means, stds = imagedatasetstats(data, C; progress = progress)
    return ImagePreprocessing(;
        means=SVector{length(means)}(means),
        stds=SVector{length(means)}(stds),
        C=C,
        kwargs...
    )
end

# Augmentation helper

"""
    augs_lighting([; intensity = 0.2, p = 0.75])

Helper to create a set of lighting transformations for image data. With
probability `p`, applies [`AdjustBrightness`](#)`(intensity)` and
[`AdjustContrast`](#)`(intensity)`.
"""
function augs_lighting(;intensity=0.2, p=0.75)
    return Maybe(AdjustBrightness(intensity), p) |> Maybe(AdjustContrast(intensity), p)
end


# Pretty-printing

function Base.show(io::IO, ip::ImagePreprocessing{P,N,C,T}) where {P,N,C,T}
    show(io, ShowTypeOf(ip))
    fields = (
        buffered = ShowLimit(ip.buffered, limit=80),
        augmentations = ShowLimit(ip.augmentations, limit=80),
    )
    show(io, ShowProps(fields, new_lines=true))
end
