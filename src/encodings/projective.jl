
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
- [`DLPipelines.Training`](#):
    1. Resizes the data so the smallest side equals
    a side length in `sz` while keeping the aspect ratio.
    2. Applies `augmentations`.
    3. Crops a random `sz`-sized portion of the data
- [`DLPipelines.Validation`](#):
    1. Resizes the data so the smallest side equals
    a side length in `sz` while keeping the aspect ratio.
    2. Crops a `sz`-sized portion from the center.
- [`DLPipelines.Inference`](#):
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
@with_kw_noshow struct ProjectiveTransforms{N} <: StatefulEncoding
    sz::NTuple{N, Int}
    buffered::Bool
    augmentations
    tfms::Dict{Context, DataAugmentation.Transform}
    sharestate::Bool
end



function ProjectiveTransforms(
        sz;
        augmentations = Identity(),
        inferencefactor = 8,
        buffered = true,
        sharestate = true)

    tfms = Dict{Context, DataAugmentation.Transform}(
        Training() => ScaleKeepAspect(sz) |> augmentations |> RandomCrop(sz) |> PinOrigin(),
        Validation() => CenterResizeCrop(sz),
        Inference() => ResizePadDivisible(sz, inferencefactor),
    )
    if buffered
        tfms[Training()] = BufferedThreadsafe(tfms[Training()])
        tfms[Validation()] = BufferedThreadsafe(tfms[Validation()])
    end

    return ProjectiveTransforms(sz, buffered, augmentations, tfms, sharestate)
end


function encodestate(enc::ProjectiveTransforms{N}, context, blocks, datas) where N
    bounds = getsamplebounds(blocks, datas, N)
    tfm = enc.tfms[context]
    randstate = DataAugmentation.getrandstate(tfm)
    return bounds, randstate
end


function encodedblock(enc::ProjectiveTransforms{N}, block::Block) where N
    return isnothing(blockitemtype(block, N)) ? nothing : block
end


function encode(
        enc::ProjectiveTransforms{N},
        context,
        block::Block,
        data;
        state=nothing) where N
    ItemType = blockitemtype(block, N)
    isnothing(ItemType) && return data
    # only init state if block is encoded
    bounds, randstate = (isnothing(state) || !enc.sharestate) ? encodestate(enc, context, block, data) : state
    # don't encode if bounds have wrong dimensionality
    bounds isa DataAugmentation.Bounds{N} || return data

    tfm = enc.tfms[context]
    item = ItemType(data, bounds)
    tdata = apply(tfm, item; randstate = randstate) |> itemdata
    return copy(tdata)
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
blockitemtype(block::Image{N}, n::Int) where N = N == n ? DataAugmentation.Image : nothing
function blockitemtype(block::Mask{N}, n::Int) where N
    return if N == n
        (data, bounds) -> DataAugmentation.MaskMulti(data, block.classes, bounds)
    else
        nothing
    end
end
blockitemtype(block::Keypoints{N}, n::Int) where N = N == n ? DataAugmentation.Keypoints : nothing
blockitemtype(block::WrapperBlock, n::Int) = blockitemtype(wrapped(block), n)


"""
    grabbounds(blocks, datas, N)

Looks through `blocks` for block data that carries `N`-dimensional
bounds information needed for projective transformations.
"""
function grabbounds(blocks::Tuple, datas::Tuple, N::Int)
    for (block, data) in zip(blocks, datas)
        bounds = grabbounds(block, data, N)
        !isnothing(bounds) && return bounds
    end
end

grabbounds(block::Image{N}, a, n) where N = N == n ? DataAugmentation.Bounds(size(a)) : nothing
grabbounds(block::Mask{N}, a, n) where N = N == n ? DataAugmentation.Bounds(size(a)) : nothing
grabbounds(block::WrapperBlock, a, n) = grabbounds(wrapped(block), a, n)


function getsamplebounds(blocks, datas, N)
    bounds = grabbounds(blocks, datas, N)
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
        flipx=true,
        flipy=false,
        max_zoom=1.5,
        max_rotate=10.,
        max_warp=0.05,
        )
    tfms = []

    flipx && push!(tfms, Maybe(FlipX()))
    flipy && push!(tfms, Maybe(FlipY()))
    max_warp > 0 && push!(tfms, WarpAffine(max_warp))
    max_rotate > 0 && push!(tfms, Rotate(max_rotate))
    push!(tfms, Zoom((1., max_zoom)))
    return DataAugmentation.compose(tfms...)
end



# Pretty-printing

function Base.show(io::IO, p::ProjectiveTransforms)
    show(io, ShowTypeOf(p))
    fields = (
        sz = ShowLimit(p.sz, limit=80),
        buffered = ShowLimit(p.buffered, limit=80),
        augmentations = ShowLimit(p.augmentations, limit=80)
    )
    show(io, ShowProps(fields, new_lines=true))
end
