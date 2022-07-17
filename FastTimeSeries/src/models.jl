
"""
blockmodel(inblock::TimeSeriesRow, outblock::OneHotTensor{0}, backbone)

Construct a model for time-series classification.
"""
function blockmodel(inblock::TimeSeriesRow,
                outblock::OneHotTensor{0}, 
                backbone)
    data   = [rand(Float32, inblock.nfeatures, 256) for _ ∈ 1:inblock.obslength]
    output = backbone(data)
    outs   = size(output)[1]
    return Models.RNNModel(backbone, outsize = length(outblock.classes), recout = outs)
end

