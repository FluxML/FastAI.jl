struct TSStats{T}
    means::AbstractArray{T,2}
    stds::AbstractArray{T,2}
end

function TSStats(means, stds)
    TSStats{eltype(means)}(means, stds)
end

"""
    TSPreprocessing() <: Encoding

Encodes 'TimeSeriesRow's by normalizing the time-series values. The time-series can
either be normalized by each variable or time-step.

Encodes
- 'TimeSeriesRow' -> 'TimeSeriesRow'
"""

struct TSPreprocessing <: Encoding
    tfms
    stats::TSStats
end

function TSPreprocessing(stats::TSStats)
    base_tfms = [
        TSStandardize
    ]
    return TSPreprocessing{}(base_tfms, stats)
end

function TSStandardize(
    obs,
    p::TSPreprocessing
)
    means = p.stats.means
    stds  = p.stats.stds
    obs = obs .- means
    obs = obs ./ stds
    return obs
end

function tsdatasetstats(
    data;
    by_var=false,
    by_step=false
)
    drop_axes = []
    if (by_var)
        append!(drop_axes,2)
    else
        append!(drop_axes,3)
    end 
    axes = [ax for ax in [1, 2, 3] if !(ax in drop_axes)]
    mean = Statistics.mean(data.table, dims=axes)
    std  = Statistics.std(data.table, dims=axes)
    return mean, std
end

function setup(::Type{TSPreprocessing}, ::TimeSeriesRow, data)
    means, stds = tsdatasetstats(data[1])
    means = reshape(means, ( size( means)[2:3] ))
    stds  = reshape(stds, ( size( stds)[2:3] ))
    stats = TSStats(means, stds)
    return TSPreprocessing(stats)
end

function encodedblock(p::TSPreprocessing, block::TimeSeriesRow)
    return block
end

function encode(p::TSPreprocessing, context, block::TimeSeriesRow, obs)
    for tfm in values(p.tfms)
        obs = tfm(obs, p)
    end
    obs
end