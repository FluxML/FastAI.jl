struct TSPreprocessing{T} <: Encoding
    means::AbstractArray{T, 2}
    stds::AbstractArray{T, 2}
    by_var
    tfms
end

function TSPreprocessing(;
    means::AbstractArray{T, 2},
    stds::AbstractArray{T, 2},
    by_var::Bool) where {T}
    tfms = [
    ]
    return TSPreprocessing{T}(means, stds, by_var, tfms)
end

function tsdatasetstats(
    data;
    by_var
)
    drop_axis = by_var ? 2 : 3
    axes = filter(!=(drop_axis), 1:3)
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

encodedblock(p::TSPreprocessing, block::TimeSeriesRow) = block

function encode(tsp::TSPreprocessing, context, block::TimeSeriesRow,  obs)
    means = tsp.means
    stds  = tsp.stds
    size(means) == size(stds) || error("`means` and `stds` must have same length")
    if (tsp.by_var)
        size(means) == (size(obs)[1], 1)
    else
        size(means) == (1, size(obs)[2])    
    end
    return ( (obs .- means) ./ stds )
end

decodedblock(p::TSPreprocessing, block::TimeSeriesRow) = block

function decode(tsp::TSPreprocessing, context, block::TimeSeriesRow,  obs)
    means = tsp.means
    stds  = tsp.stds
    #! Check size again or not.
    return ( (obs .* stds) .+ means )
end

## Tests

@testset "TimeSeriesPreprocessing [encoding]" begin

    means = stds = rand(Float32, (1,1))
    enc = TSPreprocessing(means = means, stds = stds, by_var=true)
    block = TimeSeriesRow(1, 140)
    FastAI.testencoding(enc, block)

    @testset "tsdatasetstats" begin
        
        @testset "by_var" begin
            data = cat([1 2; 3 4], [5 6; 7 8], dims=3)
            means, stds = tsdatasetstats(data, by_var=true)
            @test means[1] == 4.0
            @test stds[1] ≈ 2.5819 atol=0.001
        end

        @testset "by_step" begin
            data = cat([1 2; 3 4], [5 6; 7 8], dims=3)
            means, stds = tsdatasetstats(data, by_var=false)
            @test means[1] == 2.5
            @test stds[1] ≈ 1.2909 atol=0.001
        end

    end

    # Add Type parameter to TSPreprocessing.
    @testset "setup" begin
        data = cat([0 0; 1 1], [0 0; 1 1], dims=3)
        enc = setup(TSPreprocessing, TimeSeriesRow(2,2), data, by_var=true)
        @test enc.means[1] == 0.5
        @test enc.stds[1] ≈ 0.5773 atol=0.001
    end
end