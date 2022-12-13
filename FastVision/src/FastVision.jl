"""
    module FastVision

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
module FastVision

using FastAI
using FastAI: # blocks
              Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti, Label,
              LabelMulti, wrapped, getencodings, getblocks, encodetarget, encodeinput,
              testencoding,
# encodings
              Encoding, StatefulEncoding, OneHot,
# visualization
              ShowText,
# other
              Context, Training, Validation, Inference,
              Datasets
using FastAI.Datasets

# extending
import FastAI:
               blockmodel, blockbackbone, blocklossfn, encode, decode, checkblock,
               encodedblock, decodedblock, showblock!, mockblock, setup, encodestate,
               decodestate

import Flux
import MLUtils: getobs, numobs, mapobs, eachobs
import Colors: colormaps_sequential, Colorant, Color, Gray, Normed, RGB,
               alphacolor, deuteranopic, distinguishable_colors
using ColorVectorSpace
import FixedPointNumbers: N0f8
import DataAugmentation
import DataAugmentation: apply, Identity, ToEltype, ImageToTensor, Normalize,
                         BufferedThreadsafe, ScaleKeepAspect, PinOrigin, RandomCrop,
                         CenterResizeCrop,
                         AdjustBrightness, AdjustContrast, Maybe, FlipX, FlipY, WarpAffine,
                         Rotate, Zoom,
                         ResizePadDivisible, itemdata
import Invariants: Invariants, md, invariant, check
import ImageInTerminal
import IndirectArrays: IndirectArray
import MakieCore
import MakieCore: @recipe
import MakieCore.Observables: @map

import ProgressMeter: Progress, next!
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
include("datasets.jl")
include("recipes.jl")
include("makie.jl")

include("tests.jl")

function __init__()
    FastAI.Registries.registerrecipes(@__MODULE__, RECIPES)
    foreach(values(_tasks)) do t
        if !haskey(FastAI.learningtasks(), t.id)
            push!(FastAI.learningtasks(), t)
        end
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
