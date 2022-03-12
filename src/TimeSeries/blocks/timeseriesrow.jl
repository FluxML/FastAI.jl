"""
    TimeSeriesRow{M,N}() <: Block

`Block` for a M variate time series with N number of time steps. `obs` is valid for `TimeSeriesRow{M,N}()`
if it is an (M,N) dimensional Matrix with number element type.

## Examples

Creating a block:

```julia
TimeSeriesRow{1,51}()  # Univariate time series with length 51.
TimeSeriesRow{2,51}()  # Multivariate time series with 2 variables and length 51.
```
"""
struct TimeSeriesRow{M,N} <: Block end

function checkblock(::TimeSeriesRow{M,N}, obs::AbstractArray{T,2}) where {M,N,T<:Number}
    size(obs) == (M,N)
end

mockblock(::TimeSeriesRow{M,N}) where {M,N} = rand(Float64, (M,N))  

function setup(::Type{TimeSeriesRow}, data)
    # N,M = size(data[1,:,:])
    N, M = size(getobs(data, 1))
    return TimeSeriesRow{N,M}()
end

# visualization

function showblock!(io, ::ShowText, block::TimeSeriesRow, obs)
    plot = UnicodePlots.lineplot(obs[1,:])
    for j=2:size(obs,1)
        UnicodePlots.lineplot!(plot, obs[j,:])
    end
    print(io, plot)
end