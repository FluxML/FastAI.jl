
"""
    ImageSegmentation(classes[, sz; kwargs...]) <: LearningMethod

A learning method for image segmentation.
"""
mutable struct ImageSegmentation{N} <: DLPipelines.LearningMethod{ImageSegmentationTask}
    classes::AbstractVector
    downscale::Int
    projections::ProjectiveTransforms{N}
    imagepreprocessing::ImagePreprocessing
end


function ImageSegmentation(
        classes::AbstractVector,
        sz=(224, 224);
        aug_projection=Identity(),
        aug_image=Identity(),
        downscale=1,
        means=IMAGENET_MEANS,
        stds=IMAGENET_STDS,
        C=RGB{N0f8},
        T=Float32)

    projectivetransforms = ProjectiveTransforms(sz, augmentations=aug_projection)
    imagepreprocessing = ImagePreprocessing(;means=means, stds=stds, C=C, T=T, augmentations=aug_image)
    return ImageSegmentation(
        classes, downscale,
        projectivetransforms, imagepreprocessing)
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
    if f != 1
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
            round.(Int, size(xs)[1:end-1] ./ 2^method.downscale)...,
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


function plotxy!(f, method::ImageSegmentation, (x, y))
    image = invert(method.imagepreprocessing, x)
    mask = decodeŷ(method, Inference(), y)
    f[1, 1] = ax1 = imageaxis(f)
    f[2, 1] = ax2 = imageaxis(f)
    plotimage!(ax1, image)
    plotmask!(ax2, mask, method.classes, )
end


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
