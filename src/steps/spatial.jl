
# See **[`ProjectiveTransforms`](#).**



"""
    ProjectiveTransforms(size, [augmentations])

Pipeline step that resizes images and keypoints to `size`.

In context [`Training`](#), applies `augmentations`.
"""
@with_kw_noshow struct ProjectiveTransforms <: PipelineStep
    traintfm
    validtfm
    inferencetfm
end

function ProjectiveTransforms(
        size;
        augmentations = Identity(),
        inferencefactor = 1,
        buffered = true)
    tfms = (
        ScaleKeepAspect(size) |> augmentations |> RandomCrop(size) |> PinOrigin(),
        CenterResizeCrop(size),
        ResizePadDivisible(size, inferencefactor),
    )
    @show tfms[1]

    if buffered
        tfms = (
            BufferedThreadsafe(tfms[1]),
            BufferedThreadsafe(tfms[2]),
            # Inference transform is not buffered since it can have
            # varying sizes
            tfms[3]
        )
    end

    return ProjectiveTransforms(tfms...)
end


function Base.show(io::IO, spatial::ProjectiveTransforms)
    outsize = _parenttfm(spatial.validtfm).transforms[1].crop.size
    print(io, "ProjectiveTransforms($(outsize))")
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
        return makespatialitems(datas, makebounds(size(datas[begin])))
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


itemtype(data::AbstractMatrix{<:Colorant}) = DataAugmentation.Image
itemtype(data::Vector{<:Union{Nothing, SVector}}) = DataAugmentation.Keypoints
