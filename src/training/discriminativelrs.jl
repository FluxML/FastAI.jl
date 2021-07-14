"""
    DiscriminativeLRs(paramgroups, factors)

Use different learning rates based on `paramgroups`. `factors` maps
each group to a factor that the learning rate is multiplied by, so
for a parameter `x` the factor is
`get(factors, getgroup(paramgroups, x), 1)`.

See [`ParamGroups`](#).

## Examples

Combining with regular gradient descent, but only training a part of
the model.

```julia
using Flux.Optimise: Descent, Optimiser

model = Chain(Dense(3, 5), Dense(5, 3))
paramgroups = ParamGroups(IndexGrouper([1, 2]), model)

dlro = DiscriminativeLRs(paramgroups, Dict(1 => 0., 2 => 1.))
o = Optimiser(dlro, Descent(0.1))
```
"""
struct DiscriminativeLRs
    pg::ParamGroups
    factorfn
end

function DiscriminativeLRs(pg::ParamGroups, factors::Dict)
    return DiscriminativeLRs(pg, group -> get(factors, group, 1))
end


function apply!(o::DiscriminativeLRs, x, Δ::AbstractArray{T}) where T
    factor = convert(T, o.factorfn(getgroup(o.pg, x)))

    if factor == one(T)
        return Δ
    else
        @. Δ *= factor
    end
end


function FluxTraining.setlearningrate!(optimizer::Optimiser, value)
    FluxTraining.setlearningrate!(optimizer.os[end], value)
end
