"""
    TSPreprocessing() <: Encoding

Encodes 'TimeSeriesRow's by normalizing the time-series values. The time-series can
either be normalized by each variable or time-step.

Encodes
- 'TimeSeriesRow' -> 'TimeSeriesRow'
"""

struct TSPreprocessing <: Encoding
    tfms
end

function TSPreprocessing()
    base_tfms = [
    ]
    return TSPreprocessing(base_tfms)
end

function encodedblock(p::TSPreprocessing, block::TimeSeriesRow)
    return block
end

function encode(p::TSPreprocessing, context, block::TimeSeriesRow, obs)
    for tfm in values(p.tfms)
        obs = tfm(obs)
    end
    obs
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
    mean = Statistics.mean(data, dims=axes)
    std  = Statistics.std(data, dims=axes)
    return mean, std
end

function setup(::Type{TSPreprocessing}, ::TimeSeriesRow, data)
    means, stds = tsdatasetstats(data)
end