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
    LabelMulti, wrapped,
    # encodings
    Encoding, StatefulEncoding, OneHot,
    # visualization
    ShowText,
    # other
    FASTAI_METHOD_REGISTRY, registerlearningmethod!, Datasets
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
import FixedPointNumbers: N0f8
import DataAugmentation
import DataAugmentation: apply, Identity, ToEltype, ImageToTensor, Normalize,
    BufferedThreadsafe, ScaleKeepAspect, PinOrigin, RandomCrop, CenterResizeCrop,
    AdjustBrightness, AdjustContrast, Maybe,
    ResizePadDivisible, itemdata
import ImageInTerminal
import IndirectArrays: IndirectArray
import Requires: @require
import StaticArrays: SVector
import Statistics: mean, std
import UnicodePlots

using InlineTest


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
include("learningmethods/utils.jl")
include("learningmethods/classification.jl")
include("learningmethods/segmentation.jl")
include("learningmethods/keypointregression.jl")
include("recipes.jl")

include("tests.jl")


function __init__()
    _registerrecipes()
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
    # learning methods
    ImageClassificationSingle, ImageClassificationMulti,
    ImageKeypointRegression, ImageSegmentation

end
