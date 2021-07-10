
"""
    ScalePoints(insz) <: Encoding

Scale a `Keypoints` block falling in a rectangle of `insz` so
that they lie between -1 and 1.
"""
struct ScalePoints{N} <: Encoding
    insz::NTuple{N, Int}
end

function encode(enc::ScalePoints, context, block::Keypoints, data)
    return map(k -> (k .* (2 ./ enc.insz)) .- 1, data)
end

function decode(enc::ScalePoints, context, block::Keypoints, data)
    return map(k -> ((k) .+ 1) ./ (2 ./ enc.insz), data)
end

encodedblock(::ScalePoints, block::Keypoints{N}) where N = block
decodedblock(::ScalePoints, block::Keypoints{N}) where N = block
