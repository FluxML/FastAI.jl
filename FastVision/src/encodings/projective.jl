
"""
    ProjectiveTransforms(sz; [augmentations, buffered]) <: Encoding

Encoding for spatial data that resizes blocks to a common size `sz` and applies
projective augmentations.

Encodes all spatial blocks, preserving the block type:
- `Image{N}` -> `Image{N}`
- `Mask{N}` -> `Mask{N}`
- `Keypoints{N}` -> `Keypoints{N}`

The behavior differs based on the `context` of encoding:

{.tight}
- [`Training`](#):
    1. Resizes the data so the smallest side equals
    a side length in `sz` while keeping the aspect ratio.
    2. Applies `augmentations`.
    3. Crops a random `sz`-sized portion of the data
- [`Validation`](#):
    1. Resizes the data so the smallest side equals
    a side length in `sz` while keeping the aspect ratio.
    2. Crops a `sz`-sized portion from the center.
- [`Inference`](#):
    1. Resizes the data so the smallest side equals
    a side length in `sz` while keeping the aspect ratio.
        Note that in this context, the data does not have
        size `sz`, since no cropping happens and aspect ratio
        is preserved.


`ProjectiveTransforms` is not limited to 2D data, and works on 3D data as well.
Note, however, that some transformations in `augs_projection` (rotation, warping, flipping)
are 2D only so `augs_projection` cannot be used for 3D data.

## Keyword arguments

- `augmentations::`[`DataAugmentation.Transform`](#)` = Identity()`: Projective
    augmentation to apply during training. See [`augs_projection`](#).
- `buffered = true`: Whether to use inplace transformations. Reduces memory usage.
- `sharestate = true`: Whether to use the same random state and bounds for all blocks
    in a sample

"""
struct ProjectiveTransforms{N, T} <: StatefulEncoding
    sz::NTuple{N, Int}
    buffered::Bool
    augmentations::Any
    tfms::T
    sharestate::Bool
end

function ProjectiveTransforms(sz;
                              augmentations = Identity(),
                              inferencefactor = 8,
                              buffered = true,
                              sharestate = true)
    traintfm = ScaleKeepAspect(sz) |> augmentations |> RandomCrop(sz) |> PinOrigin()
    validtfm = CenterResizeCrop(sz)
    tfms = (;
            training = buffered ? BufferedThreadsafe(traintfm) : traintfm,
            validation = buffered ? BufferedThreadsafe(validtfm) : validtfm,
            inference = ResizePadDivisible(sz, inferencefactor))

    return ProjectiveTransforms(sz, buffered, augmentations, tfms, sharestate)
end

function encodestate(enc::ProjectiveTransforms{N}, context, blocks, obss) where {N}
    bounds = getsamplebounds(blocks, obss, N)
    tfm = _gettfm(enc.tfms, context)
    randstate = DataAugmentation.getrandstate(tfm)
    return bounds, randstate
end

_gettfm(tfms, ::Training) = tfms.training
_gettfm(tfms, ::Validation) = tfms.validation
_gettfm(tfms, ::Inference) = tfms.inference

function encodedblock(enc::ProjectiveTransforms{N}, block::Block) where {N}
    return isnothing(blockitemtype(block, N)) ? nothing : Bounded(block, enc.sz)
end

function encode(enc::ProjectiveTransforms{N},
                context,
                block::Block,
                obs;
                state = nothing) where {N}
    ItemType = blockitemtype(block, N)
    isnothing(ItemType) && return obs
    # only init state if block is encoded
    bounds, randstate = (isnothing(state) || !enc.sharestate) ?
                        encodestate(enc, context, block, obs) :
                        state
    # don't encode if bounds have wrong dimensionality
    bounds isa DataAugmentation.Bounds{N} || return obs

    tfm = _gettfm(enc.tfms, context)
    item = ItemType(obs, bounds)
    tobs = apply(tfm, item; randstate = randstate) |> itemdata
    return copy(tobs)
end

# ProjectiveTransforms is not invertible, hence no `decode` method!

# Conversion of `Block` to `DataAugmentation.Item`

"""
    blockitemtype(block, N)

Return a constructor for a `DataAugmentation.Item` that can be projected.
Return `nothing` by default, indicating that `block` cannot be turned into a
projectable item for bounds with dimensionality `N`.
For example, we have

    blockitemtype(Image{2}(), 2) -> DataAugmentation.Image

but

    blockitemtype(Image{3}(), 2) -> nothing
"""
blockitemtype(block::Block, n) = nothing
blockitemtype(block::Image{N}, n::Int) where {N} = N == n ? DataAugmentation.Image : nothing
function blockitemtype(block::Mask{N}, n::Int) where {N}
    return if N == n
        (obs, bounds) -> DataAugmentation.MaskMulti(obs, block.classes, bounds)
    else
        nothing
    end
end
function blockitemtype(block::Keypoints{N}, n::Int) where {N}
    N == n ? DataAugmentation.Keypoints : nothing
end
blockitemtype(block::WrapperBlock, n::Int) = blockitemtype(wrapped(block), n)

"""
    grabbounds(blocks, obss, N)

Looks through `blocks` for block data that carries `N`-dimensional
bounds information needed for projective transformations.
"""
function grabbounds(blocks::Tuple, obss::Tuple, N::Int)
    for (block, obs) in zip(blocks, obss)
        bounds = grabbounds(block, obs, N)
        !isnothing(bounds) && return bounds
    end
end

function grabbounds(block::Image{N}, a, n) where {N}
    N == n ? DataAugmentation.Bounds(size(a)) : nothing
end
function grabbounds(block::Mask{N}, a, n) where {N}
    N == n ? DataAugmentation.Bounds(size(a)) : nothing
end
grabbounds(block::WrapperBlock, a, n) = grabbounds(wrapped(block), a, n)

function getsamplebounds(blocks, obss, N::Int)
    bounds = grabbounds(blocks, obss, N)
    isnothing(bounds) && error("Could not detect $N-dimensional bounds needed for projective
transformations from blocks $(blocks)! Bounds can be grabbed from arrays
like `Image{$N}` block data.")
    return bounds
end

# Augmentation helper

"""
    augs_projection([; kwargs...])

Helper to create a set of projective transformations for image, mask
and keypoint data. Similar to fastai's
[`aug_transforms`](https://github.com/fastai/fastai/blob/bdc58846753c6938c63344fcaebea7149585fd5c/fastai/vision/augment.py#L946).

## Keyword arguments

- `flipx = true`: Whether to perform a horizontal flip with probability `1/2`. See [`FlipX`](#).
- `flipy = false`: Whether to perform a vertical flip with probability `1/2`. See [`FlipY`](#).
- `max_zoom = 1.5`: Maximum factor by which to zoom. Set to `1.` to disable. See [`Zoom`](#).
- `max_rotate = 10`: Maximum absolute degree by which to rotate. Set to `0.` to disable. See [`Rotate`](#).
- `max_warp = 0.05`: Intensity of corner warp. Set to `0.` to disable. See [`WarpAffine`](#).
"""
function augs_projection(;
                         flipx = true,
                         flipy = false,
                         max_zoom = 1.5,
                         max_rotate = 10.0,
                         max_warp = 0.05)
    tfms = []

    flipx && push!(tfms, Maybe(FlipX()))
    flipy && push!(tfms, Maybe(FlipY()))
    max_warp > 0 && push!(tfms, WarpAffine(max_warp))
    max_rotate > 0 && push!(tfms, Rotate(max_rotate))
    push!(tfms, Zoom((1.0, max_zoom)))
    return DataAugmentation.compose(tfms...)
end

# Pretty-printing

function Base.show(io::IO, p::ProjectiveTransforms)
    show(io, ShowTypeOf(p))
    fields = (sz = ShowLimit(p.sz, limit = 80),
              buffered = ShowLimit(p.buffered, limit = 80),
              augmentations = ShowLimit(p.augmentations, limit = 80))
    show(io, ShowProps(fields, new_lines = true))
end

# ## Tests

@testset "ProjectiveTransforms [encoding]" begin
    @testset "image" begin
        encoding = ProjectiveTransforms((32, 32))
        image = rand(RGB, 64, 96)
        block = Image{2}()

        ## We run `ProjectiveTransforms` in the different [`Context`]s:
        imagetrain = encode(encoding, Training(), block, image)
        @test size(imagetrain) == (32, 32)

        imagevalid = encode(encoding, Validation(), block, image)
        @test size(imagevalid) == (32, 32)

        imageinference = encode(encoding, Inference(), block, image)
        @test size(imageinference) == (32, 48)

        ## During inference, the aspect ratio should stay the same
        @test size(image, 1) / size(image, 2) ==
              size(imageinference, 1) / size(imageinference, 2)
    end

    @testset "keypoints" begin
        encoding = ProjectiveTransforms((32, 48))
        ks = [SVector(0.0, 0), SVector(64, 96)]
        block = Keypoints{2}(10)
        bounds = DataAugmentation.Bounds((1:32, 1:48))
        r = DataAugmentation.getrandstate(encoding.tfms.training)
        kstrain = encode(encoding, Training(), block, ks; state = (bounds, r))
        ksvalid = encode(encoding, Validation(), block, ks; state = (bounds, r))
        ksinference = encode(encoding, Inference(), block, ks; state = (bounds, r))

        @test kstrain == ksvalid == ksinference
    end

    @testset "image and keypoints" begin
        encoding = ProjectiveTransforms((32, 32))
        image = rand(RGB, 64, 96)
        ks = [SVector(0.0, 0), SVector(64, 96)]
        blocks = (Image{2}(), Keypoints{2}(10))

        @test_nowarn encode(encoding, Training(), blocks, (image, ks))
        @test_nowarn encode(encoding, Validation(), blocks, (image, ks))
        @test_nowarn encode(encoding, Inference(), blocks, (image, ks))
    end

    @testset "custom tests" begin
        enc = ProjectiveTransforms((32, 32), buffered = false)
        block = Image{2}()
        image = rand(RGB, 100, 50)

        testencoding(enc, block, image)
        @testset "randstate is shared" begin
            im1, im2 = encode(enc, Training(), (block, block), (image, image))
            @test im1 ≈ im2
        end

        @testset "don't transform data that doesn't need to be resized" begin
            imagesmall = rand(RGB, 32, 32)
            @test imagesmall ≈ encode(enc, Validation(), block, imagesmall)
        end

        @testset "3D" begin

        testencoding(ProjectiveTransforms((16, 16, 16)),
                     Image{3}(),
                     rand(RGB{N0f8}, 32, 24, 24)) end
    end

    #= depends on buffered interface
    @testset ExtendedTestSet "`run!`" begin
        encoding = ProjectiveTransforms((32, 32))
        image1 = rand(RGB, 64, 96)
        image2 = rand(RGB, 64, 96)
        buf = encode(encoding, Validation(), image1)
        cbuf = copy(buf)
        encode!(buf, encoding, Validation(), image1)
        @test buf ≈ cbuf

        encode!(buf, encoding, Validation(), image2)
        # buf should be modified on different input
        @test !(buf ≈ cbuf)
    end
    =#
end
