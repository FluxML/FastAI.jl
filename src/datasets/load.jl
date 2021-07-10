
"""
    datasetpath(name)

Return the folder that dataset `name` is stored.

If it hasn't been downloaded yet, you will be asked if you want to
download it. See [`Datasets.DATASETS`](#) for a list of available datasets.
"""
function datasetpath(name)
    datadeppath = @datadep_str "fastai-$name"
    return Path(joinpath(datadeppath, name))
end


function loadfolderdata(
        dir;
        pattern="**",
        splitfn = nothing,
        filterfn = nothing,
        loadfn = nothing)
    data = FileDataset(dir, pattern)
    @show summary(data)
    if filterfn !== nothing
        data = filterobs(filterfn, data)
    end
    if splitfn !== nothing
        data = groupobs(splitfn, data)
    end
    if loadfn !== nothing
        if splitfn === nothing
            data = mapobs(loadfn, data)
        else
            data = Dict(zip(keys(data), map(d -> mapobs(loadfn, d), values(data))))
        end
    end
    return data
end

parentname(f) = f |> pathparent |> pathname
grandparentname(f) = f |> pathparent |> pathparent |> pathname
matches(re::Regex) = f -> matches(re, f)
matches(re::Regex, f) = !isnothing(match(re, f))
const RE_IMAGEFILE = r".*\.(gif|jpe?g|tiff?|png|webp|bmp)$"i
isimagefile(f) = matches(RE_IMAGEFILE, f)



function getclassessegmentation(dir::AbstractPath)
    classes = readlines(open(joinpath(dir, "codes.txt")))
    return classes
end
getclassessegmentation(name::String) = getclassessegmentation(datasetpath(name))

#=
"""
    loadtaskdata(dir, ImageSegmentation; [split = false])

Load a data container for `ImageSegmentation` with observations
`(input = image, target = mask)`.

If `split` is `true`, returns a tuple of the data containers split by
the name of the grandparent folder.

"""
function loadtaskdata(
        dir,
        ::Type{FastAI.ImageSegmentation};
        split=false,
        kwargs...)
    imagedata = mapobs(loadfile, filterobs(isimagefile, FileDataset(joinpath(dir, "images"))))
    maskdata = mapobs(maskfromimage ∘ loadfile, filterobs(isimagefile, FileDataset(joinpath(dir, "labels"))))
    return mapobs((input = obs -> obs[1], target = obs -> obs[2]), (imagedata, maskdata))
end
=#




maskfromimage(a::AbstractArray{<:Gray{T}}) where T = maskfromimage(reinterpret(T, a))
maskfromimage(a::AbstractArray{<:Normed{T}}) where T = maskfromimage(reinterpret(T, a))
function maskfromimage(a::AbstractArray{I}) where {I<:Integer}
    return a .+ one(I)
end

loadmask(f) = f |> loadfile |> maskfromimage
