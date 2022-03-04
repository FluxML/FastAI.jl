
struct TimeSeries{M,N} <: Block end

function checkblock(::TimeSeries{M,N}, obs::AbstractArray{T,2}) where {M,N,T<:Number}
    size(obs) == (M,N)
end

mockblock(::TimeSeries{M,N}) where {M,N} = rand(Float64, (M,N))    

function setup(::Type{TimeSeries}, data)
    # N,M = size(data[1,:,:])
    N,M = size(getobs(data, 1))
    return TimeSeries{N,M}()
end

# visualization

function showblock!(io, ::ShowText, block::TimeSeries, obs)
    plot = UnicodePlots.lineplot(obs[1,:])
    for j=2:size(obs,1)
        UnicodePlots.lineplot!(plot, obs[j,:])
    end
    print(io, plot)
end
