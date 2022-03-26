
"""
    datasetpath(name)

Return the folder that dataset `name` is stored.

If it hasn't been downloaded yet, you will be asked if you want to
download it. See [`Datasets.DATASETS`](#) for a list of available datasets.
"""
function datasetpath(name)
    i = findfirst(DATASETS .== name)
    isnothing(i) && error("Dataset $name does not exist. Check `DATASETS` for available datasets.")
    config = DATASETCONFIGS[i]
    datadeppath = @datadep_str "fastai-$name"
    return Path(joinpath(datadeppath, config.subpath))
end


function loadfolderdata(
        dir;
        pattern="**",
        splitfn = nothing,
        filterfn = nothing,
        loadfn = nothing)
    data = FileDataset(dir, pattern)
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
const RE_TEXTFILE = r".*\.(txt|csv|json|md|html?|xml|yaml|toml)$"i
istextfile(f) = matches(RE_TEXTFILE, f)


maskfromimage(a::AbstractArray{<:Gray{T}}, classes) where T = maskfromimage(reinterpret(T, a), classes)
maskfromimage(a::AbstractArray{<:Normed{T}}, classes) where T = maskfromimage(reinterpret(T, a), classes)
function maskfromimage(a::AbstractArray{I}, classes) where {I<:Integer}
    a .+= one(I)
    return IndirectArray(a, classes)
end

"""
    loadmask(file, classes)

Load a segmentation mask from an image file. Returns an efficiently stored
array of type `eltype(classes)`.

"""
loadmask(file, classes) = maskfromimage(loadfile(file), classes)
