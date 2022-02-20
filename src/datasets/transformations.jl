
# mapobs

struct MappedData{F,D}
    f::F
    data::D
end

Base.show(io::IO, data::MappedData) = print(io, "mapobs($(data.f), $(summary(data.data)))")
Base.show(io::IO, data::MappedData{F,<:AbstractArray}) where {F} =
    print(io, "mapobs($(data.f), $(ShowLimit(data.data, limit=80)))")
LearnBase.nobs(data::MappedData) = nobs(data.data)
LearnBase.getobs(data::MappedData, idx::Int) = data.f(getobs(data.data, idx))
LearnBase.getobs(data::MappedData, idxs::AbstractVector) = data.f.(getobs(data.data, idxs))


"""
    mapobs(f, data)

Lazily map `f` over the observations in a data container `data`.

```julia
data = 1:10
getobs(data, 8) == 8
mdata = mapobs(-, data)
getobs(mdata, 8) == -8
```
"""
mapobs(f, data) = MappedData(f, data)
mapobs(f::typeof(identity), data) = data


"""
    mapobs(fs, data)

Lazily map each function in tuple `fs` over the observations in data container `data`.
Returns a tuple of transformed data containers.
"""
mapobs(fs::Tuple, data) = Tuple(mapobs(f, data) for f in fs)


struct NamedTupleData{TData,F}
    data::TData
    namedfs::NamedTuple{F}
end

LearnBase.nobs(data::NamedTupleData) = nobs(getfield(data, :data))

function LearnBase.getobs(data::NamedTupleData{TData,F}, idx::Int) where {TData,F}
    obs = getobs(getfield(data, :data), idx)
    namedfs = getfield(data, :namedfs)
    return NamedTuple{F}(f(obs) for f in namedfs)
end

Base.getproperty(data::NamedTupleData, field::Symbol) =
    mapobs(getproperty(getfield(data, :namedfs), field), getfield(data, :data))

Base.show(io::IO, data::NamedTupleData) =
    print(io, "mapobs($(getfield(data, :namedfs)), $(getfield(data, :data)))")

"""
    mapobs(namedfs::NamedTuple, data)

Map a `NamedTuple` of functions over `data`, turning it into a data container
of `NamedTuple`s. Field syntax can be used to select a column of the resulting
data container.

```julia
data = 1:10
nameddata = mapobs((x = sqrt, y = log), data)
getobs(nameddata, 10) == (x = sqrt(10), y = log(10))
getobs(nameddata.x, 10) == sqrt(10)
```
"""
function mapobs(namedfs::NamedTuple, data)
    return NamedTupleData(data, namedfs)
end

# filterobs

"""
    filterobs(f, data)

Return a subset of data container `data` including all indices `i` for
which `f(getobs(data, i)) === true`.

```julia
data = 1:10
nobs(data) == 10
fdata = filterobs(>(5), data)
nobs(fdata) == 5
```
"""
function filterobs(f, data; iterfn = _iterobs)
    return datasubset(data, [i for (i, obs) in enumerate(iterfn(data)) if f(obs)])
end

_iterobs(data) = [getobs(data, i) for i = 1:nobs(data)]


# groupobs

"""
    groupobs(f, data)

Split data container data `data` into different data containers, grouping
observations by `f(obs)`.

```julia
data = -10:10
datas = groupobs(>(0), data)
length(datas) == 2
```
"""
function groupobs(f, data)
    groups = Dict{Any,Vector{Int}}()
    for i = 1:nobs(data)
        group = f(getobs(data, i))
        if !haskey(groups, group)
            groups[group] = [i]
        else
            push!(groups[group], i)
        end
    end
    return Dict(group => datasubset(data, idxs) for (group, idxs) in groups)
end

# joinobs

struct JoinedData{T,N}
    datas::NTuple{N,T}
    ns::NTuple{N,Int}
end

JoinedData(datas) = JoinedData(datas, nobs.(datas))

LearnBase.nobs(data::JoinedData) = sum(data.ns)
function LearnBase.getobs(data::JoinedData, idx)
    for (i, n) in enumerate(data.ns)
        if idx <= n
            return getobs(data.datas[i], idx)
        else
            idx -= n
        end
    end
end

"""
    joinobs(datas...)

Concatenate data containers `datas`.

```julia
data1, data2 = 1:10, 11:20
jdata = joinobs(data1, data2)
getobs(jdata, 15) == 15
```
"""
joinobs(datas...) = JoinedData(datas)


# ## Tests

@testset "Data container transformations" begin
    @testset "mapobs" begin
        data = 1:10
        mdata = mapobs(-, data)
        @test getobs(mdata, 8) == -8

        mdata2 = mapobs((-, x -> 2x), data)
        @test getobs(mdata2, 8) == (-8, 16)

        nameddata = mapobs((x = sqrt, y = log), data)
        @test getobs(nameddata, 10) == (x = sqrt(10), y = log(10))
        @test getobs(nameddata.x, 10) == sqrt(10)
    end

    @testset "filterobs" begin
        data = 1:10
        fdata = filterobs(>(5), data)
        @test nobs(fdata) == 5
    end

    @testset "groupobs" begin
        data = -10:10
        datas = groupobs(>(0), data)
        @test length(datas) == 2
    end

    @testset "joinobs" begin
        data1, data2 = 1:10, 11:20
        jdata = joinobs(data1, data2)
        @test getobs(jdata, 15) == 15
    end
end
