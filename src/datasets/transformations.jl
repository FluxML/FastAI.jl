
# mapdata

struct MappedData
    f
    data
end

Base.show(io::IO, data::MappedData) = print(io, "mapdata($(data.f), $(data.data))")
LearnBase.nobs(data::MappedData) = nobs(data.data)
LearnBase.getobs(data::MappedData, idx::Int) = data.f(getobs(data.data, idx))
LearnBase.getobs(data::MappedData, idxs::AbstractVector) = data.f.(getobs(data.data, idxs))

"""
    mapdata(f, data)

Lazily map `f` over the observations in a data container `data`.

```julia
data = 1:10
getobs(data, 8) == 8
mdata = mapdata(-, data)
getobs(mdata, 8) == -8
```
"""
mapdata(f, data) = MappedData(f, data)


"""
    mapdata(fs, data)

Lazily map each function in tuple `fs` over the observations in data container `data`.
Returns a tuple of transformed data containers.
"""
mapdata(fs::Tuple, data) = Tuple(mapdata(f, data) for f in fs)

# filterdata

"""
    filterdata(f, data)

Return a subset of data container `data` including all indices `i` for
which `f(getobs(data, i)) === true`.

```julia
data = 1:10
nobs(data) == 10
fdata = filterdata(>(5), data)
nobs(fdata) == 5
```
"""
function filterdata(f, data)
    return datasubset(data, [f(getobs(data, i)) for i = 1:nobs(data)])
end


# splitdata

"""
    splitdata(f, data)

Split data container data `data` into different data containers, grouping
observations by `f(obs)`.

```julia
data = -10:10
datas = splitdata(>(0), data)
length(datas) == 2
```
"""
function splitdata(f, data)
    groups = Dict{Any, Vector{Int}}()
    for i in 1:nobs(data)
        group = f(getobs(data, i))
        if !haskey(groups, group)
            groups[group] = [i]
        else
            push!(groups[group], i)
        end
    end
    return Tuple(datasubset(data, groups[group])
        for group in sort(collect(keys(groups))))
end
