"""
    module Vision

Data blocks, encodings and more for computer vision.

The most important [`Block`] is [`Image`](#)`{N}`.


Blocks:

- [`Image`](#)`{N}`: an `N`-dimensional color image
- [`Mask`](#)`{N}`: an `N`-dimensional categorical mask
- [`Keypoints`](#)`{N}`: a fixed number of `N`-dimensional keypoints


Encodings:

- [`OneHot`](#) is implemented for `Mask`s
- [`ImagePreprocessing`](#) prepares an `Image` for training by
    normalizing and expanding the color channels
- [`KeypointPreprocessing`](#) prepares `Keypoints` for training by
    normalizing them.


"""
module Vision

using ..FastAI
using ..FastAI:
    # blocks
    Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
    LabelMulti, wrapped, getencodings, getblocks, encodetarget, encodeinput,
    # encodings
    Encoding, StatefulEncoding, OneHot,
    # visualization
    ShowText,
    # other
    Context, Training, Validation, Inference,
    Datasets
import Flux
import MLUtils: getobs, numobs, mapobs, eachobs
import FastAI.Datasets

# for tests
using ..FastAI: testencoding

# extending
import ..FastAI:
    blockmodel, blockbackbone, blocklossfn, encode, decode, checkblock,
    encodedblock, decodedblock, showblock!, mockblock, setup, encodestate,
    decodestate


import Colors: colormaps_sequential, Colorant, Color, Gray, Normed, RGB,
    alphacolor, deuteranopic, distinguishable_colors
using ColorVectorSpace
import FixedPointNumbers: N0f8
import DataAugmentation
import DataAugmentation: apply, Identity, ToEltype, ImageToTensor, Normalize,
    BufferedThreadsafe, ScaleKeepAspect, PinOrigin, RandomCrop, CenterResizeCrop,
    AdjustBrightness, AdjustContrast, Maybe, FlipX, FlipY, WarpAffine, Rotate, Zoom,
    ResizePadDivisible, itemdata
import ImageCore: colorview
import ImageInTerminal
import IndirectArrays: IndirectArray
import Invariants: invariant, md
import ProgressMeter: Progress, next!
import Requires: @require
import StaticArrays: SVector
import Statistics: mean, std
import UnicodePlots

using InlineTest
using ShowCases


# Blocks
include("blocks/bounded.jl")
include("blocks/image.jl")
include("blocks/mask.jl")
include("blocks/keypoints.jl")

include("encodings/onehot.jl")
include("encodings/imagepreprocessing.jl")
include("encodings/keypointpreprocessing.jl")
include("encodings/projective.jl")

include("models/Models.jl")
include("models.jl")

const _tasks = Dict{String, Any}()
include("tasks/utils.jl")
include("tasks/classification.jl")
include("tasks/segmentation.jl")
include("tasks/keypointregression.jl")
include("recipes.jl")

include("tests.jl")


function __init__()
    _registerrecipes()
    foreach(values(_tasks)) do t
        if !haskey(FastAI.learningtasks(), t.id)
            push!(FastAI.learningtasks(), t)
        end
    end
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        import .Makie
        import .Makie: @recipe, @lift
        import .FastAI: ShowMakie
        include("makie.jl")
    end
end

export Image, Mask, Keypoints, Bounded,
    # encodings
    ImagePreprocessing, KeypointPreprocessing, ProjectiveTransforms,
    # learning tasks
    ImageClassificationSingle, ImageClassificationMulti,
    ImageKeypointRegression, ImageSegmentation,
    # helpers
    augs_projection, augs_lighting

end
