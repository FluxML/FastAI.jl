
"""
    datasetpath(name)

Return the folder that dataset `name` is stored.

If it hasn't been downloaded yet, you will be asked if you want to
download it. See [`DATASETS`](#) for a list of available datasets.
"""
function datasetpath(name)
    datadeppath = @datadep_str "fastai-$name"
    return Path(joinpath(datadeppath, name))
end



"""
    loadtaskdata(dir, Task)
    loadtaskdata(dir, method::LearningMethod{Task})

Load a task data container for `LearningTask` `Task` stored in `dir`
in a canonical format.
"""
function loadtaskdata(dir, ::DLPipelines.LearningMethod{T}) where T
    return loadtaskdata(dir, T)
end

"""
    loadtaskdata(dir, ImageClassificationTask; [split = false])

Load a data container for `ImageClassificationTask` with observations
`(input = image, target = class)`.

If `split` is `true`, returns a tuple of the data containers split by
the name of the grandparent folder.

`dir` should contain the data in the following canonical format:

- dir
    - split (e.g. "train", "valid"...)
        - class (e.g. "cat", "dog"...)
            - image32434.{jpg/png/...}
            - ...
        - ...
    - ...
"""
function loadtaskdata(
        dir,
        Task::Type{FastAI.ImageClassificationTask};
        split=false,
        kwargs...)
    data = filterobs(isimagefile, FileDataset(dir))
    if split
        datas = groupobs(data) do path
            filename(parent(parent(obs)))
        end
        return Tuple(mapobs(
            (input = loadfile, target = path -> filename(parent(path))),
            data
        ) for data in datas)
    else
        return mapobs(
            (input = loadfile, target = path -> filename(parent(path))),
            data
        )
    end
end


"""
    getclasses(name)

Get the list of classes for classification dataset `name`.
"""
function getclassesclassification(dir)
    data = mapobs(filterobs(isimagefile, FileDataset(dir))) do path
        return filename(parent(path))
    end
    return unique(collect(eachobsparallel(data, useprimary=true, buffered=false)))
end
getclassesclassification(name::String) = getclassesclassification(datasetpath(name))


"""
    getclasses(name)

Get the list of classes for classification dataset `name`.
"""
function getclassessegmentation(dir::AbstractPath)
    classes = readlines(open(joinpath(dir, "codes.txt")))
    return classes
end
getclassessegmentation(name::String) = getclassessegmentation(datasetpath(name))

"""
    loadtaskdata(dir, ImageSegmentationTask; [split = false])

Load a data container for `ImageSegmentationTask` with observations
`(input = image, target = mask)`.

If `split` is `true`, returns a tuple of the data containers split by
the name of the grandparent folder.

"""
function loadtaskdata(
        dir,
        Task::Type{FastAI.ImageSegmentationTask};
        split=false,
        kwargs...)
    imagedata = mapobs(loadfile, filterobs(isimagefile, FileDataset(joinpath(dir, "images"))))
    maskdata = mapobs(maskfromimage ∘ loadfile, filterobs(isimagefile, FileDataset(joinpath(dir, "labels"))))
    return mapobs((input = obs -> obs[1], target = obs -> obs[2]), (imagedata, maskdata))
end




maskfromimage(a::AbstractArray{<:Gray{T}}) where T = maskfromimage(reinterpret(T, a))
maskfromimage(a::AbstractArray{<:Normed{T}}) where T = maskfromimage(reinterpret(T, a))
function maskfromimage(a::AbstractArray{I}) where {I<:Integer}
    return a .+ one(I)
end

0 - (0 - 1)
1 - (1 - 1)
10 - (10 - 1)
