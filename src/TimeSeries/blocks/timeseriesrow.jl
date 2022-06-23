"""
    TimeSeriesRow{M,N}() <: Block

[`Block`](#) for a M variate time series with N number of time steps. `obs` is valid for `TimeSeriesRow{M,N}()`
if it is an (M,N) dimensional Matrix with number element type.

## Examples

Creating a block:

```julia
TimeSeriesRow{1,51}()  # Univariate time series with length 51.
TimeSeriesRow{2,51}()  # Multivariate time series with 2 variables and length 51.
```

You can create a random observation using [`mockblock`](#):

{cell=main}
```julia
using FastAI
FastAI.mockblock(TimeSeriesRow{1,10}())
```

"""

struct TimeSeriesRow <: Block 
    nfeatures::Int
    obslength::Union{Int, Colon}
end

function checkblock(row::TimeSeriesRow, obs::AbstractArray{T,2}) where {T<:Number}
    size(obs) == (row.nfeatures, row.obslength)
end

function mockblock(row::TimeSeriesRow)
    rand(Float64, (row.nfeatures, row.obslength))
end

function setup(::Type{TimeSeriesRow}, data)
    nfeatures, obslength = size(getindex(data, 1))
    return TimeSeriesRow(nfeatures, obslength)
end
