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

To visualize a time-series sample.

```julia
showblock(TimeSeriesRow(1,10), rand(Float32, (1,10)))
```

"""

struct TimeSeriesRow <: Block 
    nfeatures::Int
    obslength::Union{Int, Colon}
end

checkblock(row::TimeSeriesRow, obs::AbstractMatrix{<:Number}) = size(obs) == (row.nfeatures, row.obslength)

function mockblock(row::TimeSeriesRow)
    rand(Float64, (row.nfeatures, row.obslength))
end

function setup(::Type{TimeSeriesRow}, data)
    nfeatures, obslength = size(getindex(data, 1))
    return TimeSeriesRow(nfeatures, obslength)
end

# visualization

function showblock!(io, ::ShowText, block::TimeSeriesRow, obs)
    plot = UnicodePlots.lineplot(obs[1,:])
    for j=2:size(obs,1)
        UnicodePlots.lineplot!(plot, obs[j,:])
    end
    print(io, plot)
end