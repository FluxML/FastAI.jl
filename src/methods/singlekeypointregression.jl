abstract type SingleKeypointRegressionTask <: DLPipelines.LearningTask end

struct SingleKeypointRegression{N} <: DLPipelines.LearningMethod{SingleKeypointRegressionTask}
    sz::NTuple{N, Int}
    projections::ProjectiveTransforms
    imagepreprocessing::ImagePreprocessing
end

# Core interface

function DLPipelines.encode(method::SingleKeypointRegression, context, sample::Union{Tuple, NamedTuple})
    image, keypoint = sample[1], sample[2]
    pimage, pkeypoints = FastAI.run(method.projections, context, (image, [keypoint]))
    x = FastAI.run(method.imagepreprocessing, context, pimage)
    y = collect(Float32.(scalepoint(pkeypoints[1], method.projections.sz)))
    return x, y
end

scalepoint(v, sz) = v .* (2 ./ sz) .- 1

# Plotting interface

_toindex(v) = CartesianIndex(Tuple(round.(Int, v)))
_boxIs(I, r) = I-(r*CartesianIndex(1, 1)):I+(r*CartesianIndex(1, 1))

function FastAI.plotxy!(f, method::SingleKeypointRegression, (x, y))
    image = FastAI.invert(method.imagepreprocessing, x)

    # draw keypoint
    v = ((y) .+ 1) ./ (2 ./ method.projections.sz)
    Is = _boxIs(_toindex(v), 1)
    for I in Is
        checkbounds(Bool, image, I) && (image[I] = RGB(1., 0, 0))
    end

    ax1 = f[1, 1] = FastAI.imageaxis(f)
    plotimage!(ax1, image)
end

# Training interface

DLPipelines.methodlossfn(method::SingleKeypointRegression) = Flux.mse

function DLPipelines.methodmodel(method::SingleKeypointRegression, backbone)
    h, w, ch, b = Flux.outdims(backbone, (method.projections.sz..., 3, 1))
    head = FastAI.Models.visionhead(ch, 2, y_range=(-1, 1))
    return Chain(backbone, head)
end
