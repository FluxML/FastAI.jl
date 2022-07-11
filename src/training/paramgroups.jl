"""
    ParamGroups(grouper, m)

A logical grouping of parameters in `m` created by [`ParamGrouper`](#)
`grouper`. Parameters in `m` are assigned a group that can be queried
using `getgroup(paramgroups, param)`. If a parameter is not assigned a
group, `getgroup` returns `nothing`.

## Examples

```julia
using Flux: Chain, Dense, params
using FastAI: ParamGroups, IndexGrouper, getgroup

model = Chain(Dense(3, 5), Dense(5, 3))
paramgroups = ParamGroups(IndexGrouper([1, 2]), model)

getgroup(paramgroups, model[1].weight) == 1
getgroup(paramgroups, model[2].weight) == 2
getgroup(paramgroups, rand(10)) === nothing
```
"""
struct ParamGroups
    map::IdDict
end

ParamGroups() = ParamGroups(IdDict())

Base.show(io::IO, ::ParamGroups) = print(io, "ParamGroups(...)")

getgroup(pg::ParamGroups, x::AbstractArray) = get(pg.map, x, nothing)

function assigngroups!(pg::ParamGroups, grouper, m)
    for (group, m_) in group(grouper, m)
        for p in Flux.params(m_)
            pg.map[p] = group
        end
    end
end

abstract type ParamGrouper end

struct IndexGrouper <: ParamGrouper
    idxs::Any
end

group(grouper::IndexGrouper, m) = Dict(i => m[is] for (i, is) in enumerate(grouper.idxs))

function ParamGroups(grouper::ParamGrouper, m)
    pg = ParamGroups()
    assigngroups!(pg, grouper, m)
    return pg
end

# ## Tests

@testset "ParamGroups" begin
    model = Chain(Dense(3, 5), Dense(5, 3))
    paramgroups = ParamGroups(IndexGrouper([1, 2]), model)

    @test getgroup(paramgroups, model[1].weight) == 1
    @test getgroup(paramgroups, model[2].weight) == 2
    @test getgroup(paramgroups, rand(10)) === nothing
end
