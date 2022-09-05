struct ContinuousPreprocessing <: Encoding
    numlabels::Int
end

ContinuousPreprocessing() = ContinuousPreprocessing(1)

decodedblock(c::ContinuousPreprocessing, block::AbstractArray) = Continuous(c.numlabels)

function encode(::ContinuousPreprocessing, _, block::Continuous, obs)
    return [obs]
end

function decode(::ContinuousPreprocessing, _, block::AbstractArray, obs)
    return obs[1]
end