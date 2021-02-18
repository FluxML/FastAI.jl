
"""
    datasetpath(name)

Return the folder that dataset `name` is stored.
If it hasn't been downloaded yet, you will be asked if you want to
download it.
"""
function datasetpath(name)
    datadeppath = @datadep_str "fastai-$name"
    return joinpath(datadeppath, name)
end


"""
    loaddataset(name)

Load dataset `name`. Check `Datasets.DATASETS` for
names of available datasets.

If it hasn't been downloaded yet, you will be asked if you want to
download it.
"""
function loaddataset(name, split = false)
    dir = datasetpath(name)
    data = filterobs(isimagefile, FileDataset(dir))
    if split
        datas = groupobs(data) do path
            filename(parent(parent(obs)))
        end
        return Tuple((
            mapobs(loadfile, data),
            mapobs(path -> filename(parent(path)), data),
        ) for data in datas)
    else
        return (
            mapobs(loadfile, data),
            mapobs(path -> filename(parent(path)), data),
        )
    end
end


function loadclasses(name)
    dir = datasetpath(name)
    data = mapobs(filterobs(isimagefile, FileDataset(dir))) do path
        return filename(parent(path))
    end
    return unique(collect(eachobsparallel(data, useprimary=true)))

end
