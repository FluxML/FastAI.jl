
# See **[`ProjectiveTransforms`](#).**



"""
    ProjectiveTransforms(sz; [augmentations, buffered])

A helper for building learning methods with vision data. Handles
resizing to `sz` and cropping images, masks and keypoints together as well
as applying projective `augmentations` like flipping and rotation.

Use with [`FastAI.run`](#), for example
`FastAI.run(::ProjectiveTransforms, context, (image, mask))`. `context`
is a [`DLPipelines.Context`](#) with different behavior for the different
contexts:

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

Can also be applied inplace to buffers of the correct size using [`FastAI.run!`](#).

Uses DataAugmentation.jl under the hood and tries to guess the [`DataAugmentation.Item`](#)
type based on the types of the data. You can also explicitly pass in the items.

`ProjectiveTransforms` is not limited to 2D data, and works on 3D data as well.
Note, however, that some transformations in `augs_projection` (rotation, warping, flipping)
are 2D only so `augs_projection` cannot be used for 3D data.

## Keyword arguments

- `augmentations::`[`DataAugmentation.Transform`](#)` = Identity()`: Projective
    augmentation to apply during training. See [`augs_projection`](#).
- `buffered = true`: Whether to use inplace transformations. Reduces memory usage.

## Examples

{cell=ProjectiveTransforms, output=false}
```julia
using FastAI, TestImages
using DLPipelines: Training, Validation, Inference

img = testimage("lighthouse")
projections = FastAI.ProjectiveTransforms((128, 128); augmentations=augs_projection())

trainimg = FastAI.run(projections, Training(), img)
```

{cell=ProjectiveTransforms, output=false}
```julia
FastAI.run(projections, Validation(), img)
```

{cell=ProjectiveTransforms, output=false}
```julia
FastAI.run(projections, Inference(), img)
```

{cell=ProjectiveTransforms, output=false}
```julia
FastAI.run!(copy(trainimg), projections, Training(), img)
```
"""
@with_kw_noshow struct ProjectiveTransforms{N} <: PipelineStep
    sz::NTuple{N, Int}
    buffered::Bool
    augmentations
    traintfm
    validtfm
    inferencetfm
end

function Base.show(io::IO, p::ProjectiveTransforms)
    show(io, ShowTypeOf(p))
    fields = (
        sz = ShowLimit(p.sz, limit=80),
        buffered = ShowLimit(p.buffered, limit=80),
        augmentations = ShowLimit(p.augmentations, limit=80)
    )
    show(io, ShowProps(fields, new_lines=true))
end

function ProjectiveTransforms(
        sz;
        augmentations = Identity(),
        inferencefactor = 1,
        buffered = true)
    tfms = (
        ScaleKeepAspect(sz) |> augmentations |> RandomCrop(sz) |> PinOrigin(),
        CenterResizeCrop(sz),
        ResizePadDivisible(sz, inferencefactor),
    )

    if buffered
        tfms = (
            BufferedThreadsafe(tfms[1]),
            BufferedThreadsafe(tfms[2]),
            # Inference transform is not buffered since it can have
            # varying sizes
            tfms[3]
        )
    end

    return ProjectiveTransforms(sz, buffered, augmentations, tfms...)
end


function run(spatial::ProjectiveTransforms, context, datas::Tuple)
    items = makespatialitems(datas)
    tfm = _gettfm(spatial, context)
    tdatas = itemdata.(DataAugmentation.apply(tfm, items))
    return deepcopy(tdatas)
end


function run!(bufs, spatial::ProjectiveTransforms, context, datas::Tuple)
    items = makespatialitems(datas)
    tfm = _gettfm(spatial, context)
    tdatas = DataAugmentation.apply(tfm, items) |> itemdata
    _copyrec!(bufs, tdatas)
    return bufs
end

run(spatial::ProjectiveTransforms, context, data) = run(spatial, context, (data,)) |> only
run!(buf, spatial::ProjectiveTransforms, context, data) = run!((buf,), spatial, context, (data,)) |> only


## Utils

_gettfm(spatial::ProjectiveTransforms, context::Training) = spatial.traintfm
_gettfm(spatial::ProjectiveTransforms, context::Validation) = spatial.validtfm
_gettfm(spatial::ProjectiveTransforms, context::Inference) = spatial.inferencetfm


function makespatialitems(datas::Tuple)
    if datas[begin] isa Item
        return makespatialitems(datas, getbounds(datas[begin]))
    else
        return makespatialitems(datas, Bounds(axes(datas[begin])))
    end
end
function makespatialitems(datas::Tuple, bounds)
    return Tuple(makeitem(data, bounds) for data in datas)
end


"""
    makeitem(data, args...)

Tries to assign a `DataAugmentation.Item` from `data` based on its type.
`args` are passed to the chosen  `Item` constructor.

- `AbstractMatrix{<:Colorant}` -> `Image`
- `Vector{<:Union{Nothing, SVector}}` -> `Keypoints`
"""
makeitem(data, args...) = itemtype(data)(data, args...)
makeitem(item::Item, args...) = item
makeitem(datas::Tuple, args...) = Tuple(makeitem(data, args...) for data in datas)


itemtype(::AbstractMatrix{<:Colorant}) = DataAugmentation.Image
itemtype(::Vector{<:Union{Nothing, SVector}}) = DataAugmentation.Keypoints
