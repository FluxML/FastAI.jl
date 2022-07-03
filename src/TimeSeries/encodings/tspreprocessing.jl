struct TSPreprocessing <: Encoding
    means
    stds
    by_var
    tfms
end

function TSPreprocessing(;
    means::AbstractArray{Float32, 2},
    stds::AbstractArray{Float32, 2},
    by_var::Bool)
    tfms = [
    ]
    return TSPreprocessing(means, stds, by_var, tfms)
end

function tsdatasetstats(
    data;
    by_var
)
    drop_axes = []
    if (by_var)
        append!(drop_axes,2)
    else
        append!(drop_axes,3)
    end 
    axes = [ax for ax in [1, 2, 3] if !(ax in drop_axes)]
    means = Statistics.mean(data, dims=axes)
    stds  = Statistics.std(data, dims=axes)
    means = reshape(means, ( size( means)[2:3] ))
    stds  = reshape(stds, ( size( stds)[2:3] ))
    return means, stds
end

function setup(::Type{TSPreprocessing}, ::TimeSeriesRow, data; by_var=true)
    means, stds = tsdatasetstats(data; by_var = by_var)
    return TSPreprocessing(means = means, stds = stds, by_var = by_var)
end

function encodedblock(p::TSPreprocessing, block::TimeSeriesRow)
    return block
end

function encode(tsp::TSPreprocessing, context, block::TimeSeriesRow,  obs)
    means = tsp.means
    stds  = tsp.stds
    size(means) == size(stds) || error("`means` and `stds` must have same length")
    if (tsp.by_var)
        size(means) == (size(obs)[1], 1)
    else
        size(means) == (1, size(obs)[2])    
    end
    obs = obs .- means
    obs = obs ./ stds
end

## Tests

# @testset "TimeSeriesPreprocessing [encoding]" begin
#     # Add Type parameter to TSPreprocessing.
#     @testset "setup" begin
#         data = cat([0 0; 1 1], [0 0; 1 1], dims=3)
#         enc = setup(TSPreprocessing, TimeSeriesRow(2,2), data, by_var=true)
#         @test enc.means[1] = 0.5
#         @test enc.stds[1] ≈ 0.5773 atol=0.001
#     end
# end