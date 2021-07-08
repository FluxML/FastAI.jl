
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

"""
@with_kw_noshow struct ProjectiveTransforms{N} <: StatefulEncoding
    sz::NTuple{N, Int}
    buffered::Bool
    augmentations
    tfms::Dict{Context, DataAugmentation.Transform}
end



function ProjectiveTransforms(
        sz;
        augmentations = Identity(),
        inferencefactor = 16,
        buffered = true)

    tfms = Dict{Context, DataAugmentation.Transform}(
        Training() => ScaleKeepAspect(sz) |> augmentations |> RandomCrop(sz) |> PinOrigin(),
        Validation() => CenterResizeCrop(sz),
        Inference() => ResizePadDivisible(sz, inferencefactor),
    )
    if buffered
        tfms[Training()] = BufferedThreadsafe(tfms[Training()])
        tfms[Validation()] = BufferedThreadsafe(tfms[Validation()])
    end

    return ProjectiveTransforms(sz, buffered, augmentations, tfms)
end


function encodestate(enc::ProjectiveTransforms{N}, context, blocks, datas) where N
    bounds = grabbounds(blocks, datas, N)
    isnothing(bounds) && error("Could not detect bounds needed for projective
        transformations from blocks $(blocks)! Bounds can be grabbed from arrays
        like `Image` block data.")
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
    bounds, randstate = isnothing(state) ? encodestate(enc, context, block, data) : state
    # don't encode if bounds have wrong dimensionality
    bounds isa DataAugmentation.Bounds{N} || return data

    tfm = enc.tfms[context]
    item = ItemType(data, bounds)
    tdatas = apply(tfm, item; randstate = randstate) |> itemdata
    return deepcopy(tdatas)
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
blockitemtype(block::Mask{N}, n::Int) where N = N == n ? DataAugmentation.MaskMulti : nothing
blockitemtype(block::Keypoints{N}, n::Int) where N = N == n ? DataAugmentation.Keypoints : nothing


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
