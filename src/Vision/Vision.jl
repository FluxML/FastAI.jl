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
    Block, WrapperBlock, AbstractBlock, OneHotTensor, OneHotTensorMulti,
    # encodings
    Encoding, StatefulEncoding, OneHot,
    # visualization
    ShowText

# extending
import ..FastAI:
    blockmodel, blockbackbone, encode, decode,
    encodedblock, decodedblock, showblock!


import Colors: colormaps_sequential, Colorant, Color, Gray, Normed
import DataAugmentation
import ImageInTerminal
import InlineTest
import StaticArrays: SVector


# Blocks
include("blocks/bounded.jl")
include("blocks/image.jl")
include("blocks/mask.jl")
include("blocks/keypoints.jl")

include("encodings/onehot.jl")
include("encodings/imagepreprocessing.jl")
include("encodings/keypointpreprocessing.jl")
include("encodings/projective.jl")

include("models.jl")


export
    # blocks
    Image, Mask, Keypoints, Bounded,

    # encodings
    ImagePreprocessing, KeypointPreprocessing, ProjectiveTransforms

end
