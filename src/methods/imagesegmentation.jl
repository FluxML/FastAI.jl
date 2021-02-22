
mutable struct ImageSegmentation{N} <: DLPipelines.LearningMethod{ImageSegmentationTask}
    sz::NTuple{N}
    classes::AbstractVector
    downscale::Int
    projectivetransforms::ProjectiveTransforms
    imagepreprocessing::ImagePreprocessing
end


function ImageSegmentation(
        classes::AbstractVector,
        sz=(224, 224);
        augmentations=Identity(),
        downscale=1,
        means=IMAGENET_MEANS,
        stds=IMAGENET_STDS,
        C=RGB{N0f8},
        T=Float32)

    projectivetransforms = ProjectiveTransforms(sz, augmentations=augmentations)
    imagepreprocessing = ImagePreprocessing(means, stds; C=C, T=T)
    return ImageSegmentation(
        sz, classes, downscale,
        projectivetransforms, imagepreprocessing)
end

DLPipelines.encode(method::ImageSegmentation, context, sample::NamedTuple) =
    DLPipelines.encode(method, context, Tuple(sample))
function DLPipelines.encode(
        method::ImageSegmentation,
        context,
        sample::Tuple)
    image, mask = sample
    imagec, maskc = run(
        method.projectivetransforms,
        context,
        (Image(image), MaskMulti(mask, method.classes)))

    x = run(method.imagepreprocessing, context, imagec)

    f = method.downscale
    if f != 1
        newsz = ntuple(i -> round(Int, size(image, i) * 1 / f), ndims(image))
        ytfm = ScaleFixed(newsz) |> DataAugmentation.Crop(newsz) |> OneHot()
    else
        ytfm = OneHot()
    end
    y = apply(ytfm, MaskMulti(maskc, method.classes)) |> itemdata
    return (x, y)
end


function DLPipelines.decodeŷ(method::ImageSegmentation, context, ŷ)
    return map(I -> I.I[end], argmax(ŷ; dims=ndims(ŷ)))
end


function DLPipelines.mocksample(method::ImageSegmentation)
    inputsz = rand.(UnitRange.(method.sz, method.sz .* 2))
    return (
        input = rand(RGB{N0f8}, inputsz),
        target = rand(1:length(method.classes), inputsz)
    )
end


function DLPipelines.mockmodel(method::ImageSegmentation)
    return function segmodel(xs)
        outsz = (
            round.(Int, size(xs)[1:end-1] ./ method.downscale)...,
            length(method.classes),
            size(xs)[end])
        return rand(Float32, outsz)
    end
end
